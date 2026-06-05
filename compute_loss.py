import torch
import torch.nn as nn
import torch.nn.functional as F
from deformable_attention import DAttention

def kl_div(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit / T, dim=-1) - F.log_softmax(q_logit / T, dim=-1)), 1)
    return torch.mean(kl) * (T * T)

class Key_Fea_Align(nn.Module):
    def __init__(self, args, student_layers, teacher_layers, student_dim, teacher_dim, s_sizes, t_sizes, device):
        super().__init__()
        self.device = device
        self.s_sizes = s_sizes
        self.t_sizes = t_sizes
        self.mapping = self._layer_mapping(student_layers, teacher_layers)
        self.evo_key_ratio = getattr(args, "evo_key_ratio", 0.2)

        self.C_attn_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(student_dim[i_s], teacher_dim[j_t], kernel_size=1, bias=False),
                nn.BatchNorm2d(teacher_dim[j_t]),
                nn.ReLU(inplace=True),
                nn.Conv2d(teacher_dim[j_t], teacher_dim[j_t], kernel_size=1, bias=False)
            ) for (i_s, j_t) in self.mapping
        ])

        self.channel_align_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(student_dim[i_s], teacher_dim[j_t], bias=False),
                nn.LayerNorm(teacher_dim[j_t]),
                nn.ReLU(inplace=True),
                nn.Linear(teacher_dim[j_t], teacher_dim[j_t], bias=False)
            ) for (i_s, j_t) in self.mapping
        ])

        self.deform_attn_student = nn.ModuleList([
            DAttention(
                channel=teacher_dim[j_t],
                q_size=self.t_sizes[j_t],
                n_heads=args.n_heads,
                n_groups=args.n_groups,
                stride=args.stride
            ).to(device)
            for (i_s, j_t) in self.mapping
        ])

        self.deform_attn_teacher = nn.ModuleList([
            DAttention(
                channel=teacher_dim[j_t],
                q_size=self.t_sizes[j_t],
                n_heads=args.n_heads,
                n_groups=args.n_groups,
                stride=args.stride
            ).to(device)
            for (i_s, j_t) in self.mapping
        ])
        self.to(device)

    def _layer_mapping(self, L_S, L_T):
        K = float(L_T) / float(L_S)
        mapping = []
        for i in range(L_S):
            j = min(int(i * K), L_T - 1)
            mapping.append((i, j))
        return mapping

    @staticmethod
    def _resize_pos_to(pos, size_hw):
        pos_chw = pos.permute(0, 3, 1, 2)
        resized = F.interpolate(pos_chw, size=size_hw, mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1)

    @staticmethod
    def _resize_score_to(score, size_hw):
        return F.interpolate(score.unsqueeze(1), size=size_hw, mode="bilinear", align_corners=False).squeeze(1)

    @staticmethod
    def _feature_score(feat):
        return feat.detach().abs().mean(dim=1)

    @staticmethod
    def _topk_count(total, ratio):
        if ratio is None or ratio <= 0:
            return total
        if 0 < ratio < 1:
            return max(1, int(total * ratio + 0.999999))
        return min(total, int(ratio))

    def _key_trajectory_loss(self, student_pos, teacher_pos, teacher_scores):
        if len(student_pos) <= 1:
            return student_pos[0].new_tensor(0.0) if student_pos else torch.tensor(0.0, device=self.device)

        evo_loss = student_pos[0].new_tensor(0.0)
        eps = 1e-8
        valid_pairs = 0

        for l in range(len(student_pos) - 1):
            p1_s, p1_t = student_pos[l], teacher_pos[l]
            p2_s, p2_t = student_pos[l + 1], teacher_pos[l + 1]
            H_l, W_l = p1_s.shape[1:3]

            p2_s_resized = self._resize_pos_to(p2_s, (H_l, W_l))
            p2_t_resized = self._resize_pos_to(p2_t, (H_l, W_l))

            phi_s = p1_s - p2_s_resized
            phi_t = p1_t - p2_t_resized

            score_l = teacher_scores[l]
            score_next = self._resize_score_to(teacher_scores[l + 1], (H_l, W_l))
            key_score = 0.5 * (score_l + score_next)

            # DAttention positions are [B * groups, H, W, 2], while feature scores are [B, H, W].
            if phi_t.shape[0] % key_score.shape[0] != 0:
                continue
            n_groups = max(1, phi_t.shape[0] // key_score.shape[0])
            key_score = key_score[:, None, :, :].expand(-1, n_groups, -1, -1).reshape(phi_t.shape[0], H_l, W_l)

            phi_s_flat = phi_s.reshape(phi_s.shape[0], -1, 2)
            phi_t_flat = phi_t.reshape(phi_t.shape[0], -1, 2)
            score_flat = key_score.reshape(key_score.shape[0], -1)

            k = self._topk_count(score_flat.shape[1], self.evo_key_ratio)
            if k <= 0:
                continue

            key_idx = torch.topk(score_flat, k=k, dim=1, largest=True).indices
            gather_idx = key_idx.unsqueeze(-1).expand(-1, -1, 2)
            phi_s_key = torch.gather(phi_s_flat, dim=1, index=gather_idx)
            phi_t_key = torch.gather(phi_t_flat, dim=1, index=gather_idx)

            phi_s_norm = phi_s_key / (phi_s_key.norm(dim=-1, keepdim=True) + eps)
            phi_t_norm = phi_t_key.detach() / (phi_t_key.detach().norm(dim=-1, keepdim=True) + eps)
            cos_sim = (phi_s_norm * phi_t_norm).sum(dim=-1)

            evo_loss += (1.0 - cos_sim).mean()
            valid_pairs += 1

        if valid_pairs == 0:
            return student_pos[0].new_tensor(0.0)
        return evo_loss / valid_pairs

    def forward(self, student_feats: list, teacher_feats: list):
        aligned_loss = 0.0
        evo_loss = 0.0

        student_pos = []
        teacher_pos = []
        teacher_scores = []

        for idx, (i_s, j_t) in enumerate(self.mapping):
            s_feat = student_feats[i_s]
            t_feat = teacher_feats[j_t]

            # Get expected channel from DAttention
            expected_channel = self.deform_attn_teacher[idx].nc

            if s_feat.dim() < 4 or t_feat.dim() < 4:
                h_t_prime = t_feat
                h_s_prime = s_feat
                if h_s_prime.shape[1] != h_t_prime.shape[1]:
                    h_s_prime = self.channel_align_fc[idx](h_s_prime)
                t_pos = None
                s_pos = None
            else:
                # Ensure t_feat has correct number of channels
                if t_feat.shape[1] != expected_channel:
                    if t_feat.shape[1] > expected_channel:
                        # Truncate extra channels (e.g., extra positional encoding)
                        t_feat = t_feat[:, :expected_channel, :, :]
                    else:
                        # Pad with zeros if fewer channels
                        pad_size = expected_channel - t_feat.shape[1]
                        t_feat = F.pad(t_feat, (0, 0, 0, 0, 0, pad_size))

                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
                s_proj = self.C_attn_list[idx](s_feat)

                # Ensure s_proj has correct number of channels
                if s_proj.shape[1] != expected_channel:
                    if s_proj.shape[1] > expected_channel:
                        s_proj = s_proj[:, :expected_channel, :, :]
                    else:
                        pad_size = expected_channel - s_proj.shape[1]
                        s_proj = F.pad(s_proj, (0, 0, 0, 0, 0, pad_size))

                h_t_prime, t_pos = self.deform_attn_teacher[idx](t_feat)
                h_s_prime, s_pos = self.deform_attn_student[idx](s_proj)

            aligned_loss += F.mse_loss(h_s_prime, h_t_prime)

            if t_pos is not None or s_pos is not None:
                teacher_pos.append(t_pos)
                student_pos.append(s_pos)
                teacher_scores.append(self._feature_score(h_t_prime))

        if len(student_pos) > 1:
            evo_loss = self._key_trajectory_loss(student_pos, teacher_pos, teacher_scores)

        return aligned_loss, evo_loss

class student_distillationloss(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.fea_align = None

    def _init_modules(self, student_feats, teacher_feats):
        student_dim = [f.shape[1] for f in student_feats]
        teacher_dim = [f.shape[1] for f in teacher_feats]

        self.s_sizes = [
            (f.shape[2], f.shape[3]) if f.dim() == 4 else (1, 1)
            for f in student_feats
        ]

        self.t_sizes = [
            (f.shape[2], f.shape[3]) if f.dim() == 4 else (1, 1)
            for f in teacher_feats
        ]

        self.fea_align = Key_Fea_Align(
            args=self.args,
            student_layers=len(student_feats),
            teacher_layers=len(teacher_feats),
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            s_sizes=self.s_sizes,
            t_sizes=self.t_sizes,
            device=self.device
        )

    def forward(self, student_feats, student_logits, teacher_feats, teacher_logits, labels):
        if self.fea_align is None:
            self._init_modules(student_feats, teacher_feats)

        labels = labels.long().to(student_logits.device)
        task_loss = F.cross_entropy(student_logits, labels)

        output_loss = kl_div(teacher_logits.detach(), student_logits, self.args.temperature)
        aligned_loss, evo_loss = self.fea_align(student_feats, [f.detach() for f in teacher_feats])

        if not getattr(self.args, "use_aligned_loss", True):
            aligned_loss = 0.0

        if not getattr(self.args, "use_evo_loss", True):
            evo_loss = 0.0

        total_loss = task_loss + output_loss + self.args.alpha * aligned_loss + self.args.beta * evo_loss
        return total_loss

class teacher_distillationloss(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.temperature = getattr(args, "temperature", 1.0)
        self.args = args

    def forward(self, teacher_logits, student_logits, labels):
        teacher_loss = F.cross_entropy(teacher_logits, labels)
        kd_loss = kl_div(student_logits.detach(), teacher_logits, self.temperature)

        loss = teacher_loss + kd_loss
        return loss
