import onnx

# import netron


# onnx_model_path = "simplify_petr.onnx"  # 替换为你的 ONNX 文件路径
# netron.start(onnx_model_path)  # 在浏览器中打开 ONNX 模型可视化界面


def simple_onnx_operator_analysis(onnx_model_path):
    """
    简单分析 ONNX 模型算子：统计种类
    """
    # 1. 加载 ONNX 模型
    model = onnx.load(onnx_model_path)

    # 2. 验证模型合法性（可选，确保模型无错误）
    try:
        onnx.checker.check_model(model)
        print("✅ ONNX 模型合法")
    except Exception as e:
        print("❌ ONNX 模型不合法：", e)
        return

    # 3. 提取所有算子节点（node）
    print("\n" + "=" * 50)
    print("📋 所有算子节点详情（类型 + 名称）：")
    print("=" * 50)
    operator_types = []  # 用于收集所有算子类型，后续去重统计

    for idx, node in enumerate(model.graph.node):
        # node.op_type：算子类型（如 IsNaN、Where、Constant）
        # node.name：算子节点名称（自动生成，可用于区分同一类型的不同节点）
        print(f"序号 {idx+1} | 算子类型：{node.op_type:10} | 节点名称：{node.name}")
        operator_types.append(node.op_type)

    # 4. 去重统计模型中用到的算子种类
    print("\n" + "=" * 50)
    print("📊 模型中用到的算子种类（去重后）：")
    print("=" * 50)
    unique_operators = sorted(list(set(operator_types)))
    for op in unique_operators:
        count = operator_types.count(op)
        print(f"• {op}（出现 {count} 次）")


def detailed_onnx_operator_analysis(onnx_model_path):
    """
    详细分析 ONNX 模型算子：统计次数 + 定位每个算子的具体位置/详情
    """
    # 1. 加载 ONNX 模型并验证合法性
    try:
        model = onnx.load(onnx_model_path)
        onnx.checker.check_model(model)
        print("✅ ONNX 模型合法")
    except Exception as e:
        print(f"❌ ONNX 模型不合法或加载失败：{e}")
        return

    # 2. 初始化数据结构：存储算子详情（按序号）和分组信息（按算子类型）
    operator_full_details = []  # 存储所有算子的完整详情，格式：(序号, 类型, 名称, 输入张量, 输出张量)
    operator_grouped = {}  # 按算子类型分组，格式：{op_type: [节点1详情, 节点2详情, ...]}

    # 3. 遍历所有算子节点，提取完整信息
    graph = model.graph
    print(f"\n📌 模型计算图名称：{graph.name if graph.name else '默认计算图'}")
    print("\n" + "=" * 80)
    print("📋 所有算子节点完整详情（按执行顺序）：")
    print("=" * 80)
    for idx, node in enumerate(graph.node):
        # 提取关键信息：序号、算子类型、节点名称、输入张量、输出张量
        op_idx = idx + 1
        op_type = node.op_type
        op_name = node.name if node.name else f"{op_type}_{op_idx}"
        op_inputs = [inp for inp in node.input]  # 该算子的输入张量列表
        op_outputs = [out for out in node.output]  # 该算子的输出张量列表

        # 存储到完整详情列表
        node_detail = (op_idx, op_type, op_name, op_inputs, op_outputs)
        operator_full_details.append(node_detail)

        # 按算子类型分组存储
        if op_type not in operator_grouped:
            operator_grouped[op_type] = []
        operator_grouped[op_type].append(node_detail)

        # 打印当前节点详情
        print(f"\n序号 {op_idx}")
        print(f"  算子类型：{op_type}")
        print(f"  节点名称：{op_name}")
        print(f"  输入张量：{op_inputs if op_inputs else '无'}")
        print(f"  输出张量：{op_outputs if op_outputs else '无'}")

    # 4. 按算子类型分组汇总（统计次数 + 定位具体位置）
    print("\n" + "=" * 80)
    print("📊 算子类型分组汇总（次数 + 具体位置）：")
    print("=" * 80)
    # 按算子类型排序输出
    for op_type in sorted(operator_grouped.keys()):
        node_list = operator_grouped[op_type]
        op_count = len(node_list)
        print(f"\n🔹 算子类型：{op_type}（共出现 {op_count} 次）")
        print(f"  出现位置/节点详情：")
        for node_detail in node_list:
            op_idx, _, op_name, op_inputs, op_outputs = node_detail
            print(f"    - 序号 {op_idx} | 节点名称：{op_name} | 输入：{op_inputs[:2]}...（完整列表见上方）")


# 调用函数进行分析（替换为你的 ONNX 文件路径）
if __name__ == "__main__":
    # onnx_model_path = "work_dirs/petr/simplify_petr.onnx"
    onnx_model_path = "work_dirs/3dppe/simplify_3dppe.onnx"
    simple_onnx_operator_analysis(onnx_model_path)
    # detailed_onnx_operator_analysis(onnx_model_path)
