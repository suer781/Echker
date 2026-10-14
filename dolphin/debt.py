"""双半球债务账本（胼胝体通道）。

原理→代码映射：胼胝体是两半球信息传递通道（裂脑→认知割裂）；
映射为 DebtLedger——债务互相结算（耦合）+ 蒸馏方向交替（互为师生）+ KL 健康度观测。
"""


class DebtLedger:
    def __init__(self):
        self.debts = [0.0, 0.0]     # 每半球欠对方的训练债务（字节当量）
        self.notes = [[], []]       # 蒸馏笔记 eid 列表（胼胝体通道载荷）
        self.callosum_kl = None     # 最近一次两半球输出分布 KL（通道健康度）

    def to_state(self):
        return {"debts": self.debts, "notes": self.notes, "callosum_kl": self.callosum_kl}

    def from_state(self, st):
        if not st:
            return self
        self.debts = [float(x) for x in st.get("debts", [0.0, 0.0])]
        self.notes = [list(x) for x in st.get("notes", [[], []])]
        self.callosum_kl = st.get("callosum_kl")
        return self

    def settle(self):
        """结算：合并未消化债并返回结算记录（S5 接口：含 settled_debt/notes_exchanged）。"""
        rec = {"before": list(self.debts)}
        settled = 0.0
        # 债务归一化：若两半球债务都小则清零，避免长期残留
        if all(d < 10.0 for d in self.debts):
            settled = sum(self.debts)
            self.debts = [0.0, 0.0]
        rec["after"] = list(self.debts)
        # spec 要求字段：本次结算的债务量；本次交换的笔记数（初版恒 0）。
        rec["settled_debt"] = float(settled)
        rec["notes_exchanged"] = 0
        return rec