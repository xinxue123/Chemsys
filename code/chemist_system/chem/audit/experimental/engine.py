#!/usr/bin/env python3
import numpy as np
import torch
from .physics import get_Phi_FU
from pymatgen.core import Composition
import os
import time

kB = 8.617333262e-5  # eV/K
device = torch.device('cuda:0')
def force_add_path(engine, reactant_names, product_names):
    """强制在 S_gpu 矩阵中注入一条特定的化学反应路径"""
    new_col = np.zeros(engine.N)
    for name in reactant_names:
        new_col[engine.names.index(name)] = -1.0
    for name in product_names:
        # 假设 1 + 1 -> 2 的比例，这里需要根据你的 FU 调整系数
        new_col[engine.names.index(name)] = 2.0 
    
    # 更新 GPU 矩阵
    new_path = torch.tensor(new_col, device=device).unsqueeze(1)
    engine.S_gpu = torch.cat([engine.S_gpu, new_path], dim=1)
    # print(f"成功注入路径: {' + '.join(reactant_names)} -> {' + '.join(product_names)}")
class KineticEngine:
    def __init__(self, phases, T=1000, P=0.21):
        self.phases = phases
        self.T = T
        self.P = P
        self.N = len(phases)

        self.names = [Composition(p.name).reduced_formula for p in phases]
        self.phase_idx = {}
        for i, name in enumerate(self.names):
            if name not in self.phase_idx:
                self.phase_idx[name] = i

        # Build composition matrix (elements x phases)
        all_els = set()
        for p in self.phases:
            for el in p.composition.keys():
                all_els.add(str(el))
        self.elements = sorted(list(all_els))
        n_el = len(self.elements)

        A_np = np.zeros((n_el, self.N))
        for i, p in enumerate(self.phases):
            cd = {str(k): v for k, v in p.composition.items()}
            for e_idx, el in enumerate(self.elements):
                A_np[e_idx, i] = cd.get(el, 0)

        self.A_np = A_np
        self.A_gpu = torch.tensor(A_np, dtype=torch.float64, device=device)

        # print(f"  [Init] {self.N} phases, {n_el} elements: {self.elements}")
        # print(f"  [GPU] Using {torch.cuda.get_device_name(0)}")
        self._build_network_gpu()

    def _build_network_gpu_v1(self):
        """Build reaction network using GPU-accelerated vectorized ops."""
        cache_path = f"S_gpu_{self.N}.npy"
        if os.path.exists(cache_path):
            S_np = np.load(cache_path)
            self.S_gpu = torch.tensor(S_np, dtype=torch.float64, device=device)
            print(f"  [Engine] Loaded cached network: {S_np.shape[1]} paths")
            return

        t0 = time.time()
        print("  [Init] Building reaction network on GPU...")
        A = self.A_gpu  # (n_el, N)
        N = self.N
        n_el = A.shape[0]

        # Generate all (i, j, k) triples where i < j, k != i, k != j
        S_list = []

        # For each product k, try all pairs (i, j) as reactants
        # Vectorize: for each k, build all i,j pairs at once
        pairs_i = []
        pairs_j = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs_i.append(i)
                pairs_j.append(j)
        pairs_i = torch.tensor(pairs_i, device=device)
        pairs_j = torch.tensor(pairs_j, device=device)
        n_pairs = len(pairs_i)

        # A columns for all pairs
        Ai_all = A[:, pairs_i]  # (n_el, n_pairs)
        Aj_all = A[:, pairs_j]  # (n_el, n_pairs)

        for k in range(N):
            Ak = A[:, k]  # (n_el,)

            # Filter out pairs containing k
            mask = (pairs_i != k) & (pairs_j != k)
            idx = torch.where(mask)[0]
            if len(idx) == 0:
                continue

            Ai = Ai_all[:, idx]  # (n_el, n_valid)
            Aj = Aj_all[:, idx]  # (n_el, n_valid)
            pi = pairs_i[idx]
            pj = pairs_j[idx]

            # Build 3x3 determinant for coplanarity check
            # det([Ai, Aj, Ak]) for each valid pair
            # For 3x3: det = a(ei-fh) - b(di-fg) + c(dh-eg)
            if n_el == 3:
                a, b, c = Ai[0], Aj[0], Ak[0]
                d, e, f = Ai[1], Aj[1], Ak[1]
                g, h, ii = Ai[2], Aj[2], Ak[2]
                det = a * (e * ii - f * h) - b * (d * ii - f * g) + c * (d * h - e * g)
                coplanar = torch.abs(det) < 1e-6
            else:
                # General case: stack and compute det
                mats = torch.stack([Ai.T, Aj.T, Ak.unsqueeze(0).expand_as(Ai.T)], dim=2)
                det = torch.linalg.det(mats)
                coplanar = torch.abs(det) < 1e-6

            cp_idx = torch.where(coplanar)[0]
            if len(cp_idx) == 0:
                continue

            # For coplanar pairs, solve least-squares: [Ai, Aj] @ coeffs = Ak
            Ai_cp = Ai[:, cp_idx]  # (n_el, n_cp)
            Aj_cp = Aj[:, cp_idx]  # (n_el, n_cp)
            pi_cp = pi[cp_idx]
            pj_cp = pj[cp_idx]

            # Batch lstsq: solve for each pair
            # mat = (n_cp, n_el, 2), rhs = (n_cp, n_el)
            mat_batch = torch.stack([Ai_cp.T, Aj_cp.T], dim=2)  # (n_cp, n_el, 2)
            rhs_batch = Ak.unsqueeze(0).expand(len(cp_idx), -1)  # (n_cp, n_el)

            try:
                result = torch.linalg.lstsq(mat_batch, rhs_batch.unsqueeze(-1))
                coeffs = result.solution.squeeze(-1)  # (n_cp, 2)

                # Check residual
                pred = torch.bmm(mat_batch, coeffs.unsqueeze(-1)).squeeze(-1)
                residuals = torch.norm(pred - rhs_batch, dim=1)

                # Filter: good residual and positive coefficients
                good = (residuals < 1e-10) & (coeffs[:, 0] > 1e-4) & (coeffs[:, 1] > 1e-4)
                good_idx = torch.where(good)[0]

                for g in good_idx:
                    s = np.zeros(N)
                    s[pi_cp[g].item()] = -coeffs[g, 0].item()
                    s[pj_cp[g].item()] = -coeffs[g, 1].item()
                    s[k] = 1.0
                    S_list.append(s)
            except:
                continue

        # Polymorph transitions
        for i in range(N):
            for j in range(i + 1, N):
                if torch.norm(A[:, i] - A[:, j]).item() < 1e-10:
                    s = np.zeros(N)
                    s[i], s[j] = -1.0, 1.0
                    S_list.append(s)

        if S_list:
            S_np = np.array(S_list).T
        else:
            S_np = np.zeros((N, 0))

        np.save(cache_path, S_np)
        self.S_gpu = torch.tensor(S_np, dtype=torch.float64, device=device)
        elapsed = time.time() - t0
        print(f"  [Engine] Built {S_np.shape[1]} paths in {elapsed:.1f}s (GPU)")
    def _build_network_gpu(self):
        """
        构建反应网络：支持 2-to-1, 1-to-2, 和 2-to-2 (复分解) 反应模式。
        使用 GPU 向量化操作加速海量组合的原子守恒校验。
        """
        # 1. 缓存检查：避免重复计算 103x103 的庞大网络
        cache_path = f"S_gpu_full_network_{self.N}.npy"
        if os.path.exists(cache_path):
            S_np = np.load(cache_path)
            self.S_gpu = torch.tensor(S_np, dtype=torch.float64, device=device)
            # print(f"  [Engine] 加载缓存网络: {S_np.shape[1]} 条路径 (含 4-Phase)")
            return

        t0 = time.time()
        # print(f"  [Init] 正在构建 {self.N} 个物相的全自动反应网络...")
        
        A = self.A_gpu  # (n_el, N)
        N = self.N
        S_list = []

        # 预生成所有可能的反应物对 (i, j)
        pairs_i, pairs_j = torch.triu_indices(N, N, offset=1, device=device)
        n_pairs = len(pairs_i)
        
        # 模式 A: 搜索 2-to-1 和 1-to-2 (3-Phase)
        # 逻辑：A_i + A_j = A_k
        # print("    > 正在搜索 3-Phase 路径 (A + B <-> C)...")
        for k in range(N):
            mask = (pairs_i != k) & (pairs_j != k)
            idx = torch.where(mask)[0]
            if len(idx) == 0: continue
            
            # 计算残差：(A_i + A_j) - A_k
            res = (A[:, pairs_i[idx]] + A[:, pairs_j[idx]]) - A[:, k].unsqueeze(1)
            dist = torch.norm(res, dim=0)
            
            match_idx = torch.where(dist < 1e-6)[0]
            for m in match_idx:
                s = np.zeros(N)
                s[pairs_i[idx][m].item()] = -1.0
                s[pairs_j[idx][m].item()] = -1.0
                s[k] = 1.0
                S_list.append(s)

        # 模式 B: 搜索 2-to-2 (4-Phase) - 核心修复点
        # 逻辑：A_i + A_j = A_k + A_m  =>  (A_i + A_j) - A_k = A_m
        # print("    > 正在搜索 4-Phase 路径 (A + B <-> C + D)...")
        for k in range(N):
            # 这里的 A_sum 是所有 (i,j) 对的原子总和
            A_sum = A[:, pairs_i] + A[:, pairs_j]  # (n_el, n_pairs)
            
            # 计算剩余项：Sum - A_k
            remainder = A_sum - A[:, k].unsqueeze(1) # (n_el, n_pairs)
            
            # 在全库中寻找是否存在物相匹配这个 remainder
            # 使用高效率的矩阵比对
            for m in range(k + 1, N): # 避免重复搜索 (C,D) 和 (D,C)
                diff = torch.norm(remainder - A[:, m].unsqueeze(1), dim=0)
                match_pairs = torch.where(diff < 1e-6)[0]
                
                for p_idx in match_pairs:
                    p1, p2 = pairs_i[p_idx].item(), pairs_j[p_idx].item()
                    if m == p1 or m == p2 or k == p1 or k == p2: continue
                    
                    s = np.zeros(N)
                    s[p1], s[p2] = -1.0, -1.0
                    s[k], s[m] = 1.0, 1.0
                    S_list.append(s)

        # 模式 C: 搜索同质异构体转变 (Polymorphs)
        # print("    > 正在搜索相变路径 (A <-> A')...")
        for i in range(N):
            for j in range(i + 1, N):
                if torch.norm(A[:, i] - A[:, j]) < 1e-8:
                    s = np.zeros(N)
                    s[i], s[j] = -1.0, 1.0
                    S_list.append(s)

        # # print("    > 正在注入手动指定的关键路径...")
        # manual_reactions = [
        #     # 格式: ([反应物名称列表], [产物名称列表], [反应物系数], [产物系数])
        #     (['Ba2TiO4', 'TiO2'], ['BaTiO3'], [1.0, 1.0], [2.0]),
        #     # 你可以继续在这里添加其他无法自动搜到的反应
        # ]

        # for reactants, products, r_coeffs, p_coeffs in manual_reactions:
        #     try:
        #         s = np.zeros(N)
        #         # 填充反应物 (负值)
        #         for name, coeff in zip(reactants, r_coeffs):
        #             s[self.names.index(name)] = -float(coeff)
        #         # 填充产物 (正值)
        #         for name, coeff in zip(products, p_coeffs):
        #             s[self.names.index(name)] = float(coeff)
                
        #         S_list.append(s)
        #         # print(f"      [Manual] 已注入路径: {' + '.join(reactants)} -> {' + '.join(products)}")
        #     except ValueError as e:
        #         pass
        #         # print(f"      [Warning] 手动注入失败: {e}")


        # 4. 汇总与去重
        if S_list:
            # 转换为矩阵并去除重复的化学路径
            S_np = np.array(S_list)
            S_np = np.unique(S_np, axis=0).T
        else:
            S_np = np.zeros((N, 0))

        np.save(cache_path, S_np)
        self.S_gpu = torch.tensor(S_np, dtype=torch.float64, device=device)
        elapsed = time.time() - t0
        # print(f"  [Engine] 构建完成: 发现 {S_np.shape[1]} 条守恒路径，耗时 {elapsed:.1f}s")

    def evolve_at_T(self, c0_np, T, max_steps=5000, k_base=1.0):
        """GPU-accelerated evolution at fixed T with strict mass conservation."""
        # self.debug_network()
        Phi_np = np.array([get_Phi_FU(p, T) for p in self.phases])
        Phi = torch.tensor(Phi_np, dtype=torch.float64, device=device)

        S = self.S_gpu  # (N, M)
        M = S.shape[1]
        if M == 0:
            return c0_np.copy()

        S_T = S.T  # (M, N)
        # # 原始版本的 Arrhenius 速率常数 (现在我们将其替换为基于 BEP 的动态速率)
        Ea = 1.0  # eV
        arrhenius = np.exp(-Ea / (kB * T)) if T > 10 else 0.0


        # Classify reactions on GPU
        dG = S_T @ Phi  # (M,)
        abs_stoich_sum = torch.sum(torch.abs(S_T), dim=1)
        n_mol = torch.clamp(abs_stoich_sum / 2.0, min=1.0)
        dG_n = dG / n_mol

        rate_gpu = torch.zeros(M, dtype=torch.float64, device=device)
        dir_gpu = torch.zeros(M, dtype=torch.float64, device=device)


        # 1. 定义 BEP 参数 (通常 alpha 在 0.4 到 0.6 之间)
        Ea_base = 1.25  # 基础能垒 eV (作为“起步价”)
        alpha = 0.5     # 线性系数
        # 2. 计算每个反应方向的活化能向量 (M,)
        # 核心逻辑：dG_n 越负（释放能量越多），Ea 越低
        # 我们只针对“顺流”方向降低能垒
        Ea_fwd = Ea_base + alpha * dG_n # 如果 dG_n 是负的，Ea 减小
        Ea_bwd = Ea_base - alpha * dG_n # 逆向反应，dG_n 是正的，Ea 减小
        
        # 物理限制：Ea 不能为负数，设置一个最小阈值 (比如 0.05 eV)
        Ea_fwd = torch.clamp(Ea_fwd, min=0.05)
        Ea_bwd = torch.clamp(Ea_bwd, min=0.05)

        # 3. 计算向量化的阿伦尼乌斯项 (不再是一个固定的 arrhenius)
        # 注意：这里不再使用单一的 np.exp，而是使用 torch.exp 作用于整个向量
        arrhenius_fwd = torch.exp(-Ea_fwd / (kB * T))
        arrhenius_bwd = torch.exp(-Ea_bwd / (kB * T))

        fwd = dG_n < -1e-6
        bwd = dG_n > 1e-6
        rate_gpu[fwd] = torch.abs(dG_n[fwd]) * arrhenius_fwd[fwd] * k_base
        # rate_gpu[fwd] = torch.abs(dG_n[fwd]) * arrhenius * k_base
        dir_gpu[fwd] = 1.0
        rate_gpu[bwd] = torch.abs(dG_n[bwd]) * arrhenius_bwd[bwd] * k_base
        # rate_gpu[fwd] = torch.abs(dG_n[fwd]) * arrhenius * k_base
        dir_gpu[bwd] = -1.0


        # Precompute reactant coefficient matrix
        R_coeff = torch.zeros(M, self.N, dtype=torch.float64, device=device)
        for r in range(M):
            if dir_gpu[r] > 0:
                mask = S_T[r] < -1e-8
                R_coeff[r, mask] = torch.abs(S_T[r, mask])
            elif dir_gpu[r] < 0:
                mask = S_T[r] > 1e-8
                R_coeff[r, mask] = torch.abs(S_T[r, mask])

        c = torch.tensor(c0_np, dtype=torch.float64, device=device)
        base_dt = 5e-3
        active_mask = rate_gpu > 1e-15

        for step in range(max_steps):
            c_safe = torch.clamp(c, min=0.0)

            # Vectorized limiting reagent: c_safe / R_coeff, masked
            # Where R_coeff is 0, set ratio to inf
            ratio = torch.where(R_coeff > 1e-10, c_safe.unsqueeze(0) / R_coeff, torch.tensor(1e30, device=device))
            avail = torch.min(ratio, dim=1).values  # (M,)
            avail[~active_mask] = 0.0

            # Path fluxes
            pf = torch.zeros(M, dtype=torch.float64, device=device)

            can_react = (avail > 1e-12) & active_mask
            pf[can_react] = dir_gpu[can_react] * rate_gpu[can_react] * avail[can_react]

            net_flux = S @ pf  # (N,)

            # Convergence check
            if torch.max(torch.abs(net_flux)).item() < 1e-8:
                break

            # Safe dt: no species goes negative
            dt = base_dt
            depleting = (net_flux < -1e-12) & (c_safe > 1e-16)
            if depleting.any():
                safe_dts = c_safe[depleting] / torch.abs(net_flux[depleting])
                min_safe = torch.min(safe_dts).item()
                if min_safe < dt:
                    dt = min_safe * 0.90

            c = c + net_flux * dt
            c = torch.clamp(c, min=0.0)

        return c.cpu().numpy()