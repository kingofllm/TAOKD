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

    def forward(self, student_feats: list, teacher_feats: list):
        aligned_loss = 0.0
        evo_loss = 0.0

        student_pos = []
        teacher_pos = []

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

        if len(student_pos) > 1:
            evo_loss = 0.0
            eps = 1e-8

            for l in range(len(student_pos) - 1):
                p1_s, p1_t = student_pos[l], teacher_pos[l]
                p2_s, p2_t = student_pos[l + 1], teacher_pos[l + 1]
                H_l, W_l = p1_s.shape[1:3]

                p2_s_resized = F.interpolate(p2_s.permute(0, 3, 1, 2), size=(H_l, W_l),
                                     mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                p2_t_resized = F.interpolate(p2_t.permute(0, 3, 1, 2), size=(H_l, W_l),
                                     mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

                phi_s = p1_s - p2_s_resized
                phi_t = p1_t - p2_t_resized

                phi_s_flat = phi_s.reshape(phi_s.shape[0], -1, 2)
                phi_t_flat = phi_t.reshape(phi_t.shape[0], -1, 2)

                phi_s_norm = phi_s_flat / (phi_s_flat.norm(dim=-1, keepdim=True) + eps)
                phi_t_norm = phi_t_flat / (phi_t_flat.norm(dim=-1, keepdim=True) + eps)

                cos_sim = (phi_s_norm * phi_t_norm).sum(dim=-1)

                evo_loss += (1.0 - cos_sim).mean()

            evo_loss /= (len(student_pos) - 1)

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