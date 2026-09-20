# 孙绍齐's CV

- Phone: +86 177 2483 8277
- Email: [shaoqisun@qq.com](mailto:shaoqisun@qq.com)
- Website: [shao12138.blog.csdn.net](https://shao12138.blog.csdn.net/)


# 教育背景
## **国防科技大学（985，双一流高校）**, 软件工程（A+，双一流学科）

**硕士**

湖南-长沙

2021 – 2024

- 方向：多智能体强化学习（MARL）

- 论文：基于策略差异化的多智能体内在奖励研究（IEEE Trans*2、CCF-B、CCF-C）

- 博客：CSDN 访问量 220w；LeetCode：716



## **郑州大学（211，双一流高校）**, 软件工程

**学士**

河南-郑州

2017 – 2021

- 方向：自然语言处理（NLP）

- 论文：基于交互式多任务学习的中文文本情感分类（SCI）

- 绩点：3.63/4.00；排名：12/712；英语：CET-6



# 工作经验
## **微众银行 | 武汉研发中心 | 数据算法岗**

湖北-武汉

2024 – 至今

- 方向：大语言模型微调（Fine-tune）、文本转语音算法（TTS）推理加速

- 项目：AgentFlywheel（营销场景）、DeepReview、AOB 智能外呼系统（TTS推理加速）

- 专利：MARL 下基于影响传播的奖励冲突解决方法、一种自适应波动流量下限流熔断策略



# 项目经历
## **AgentFlywheel（微粒贷营销数据飞轮）**

9月 2026

由造数、微调、对话、质检与分析五大 Agent 闭环联动，实现“造数→微调→对话→质检→分析→回流造数”的自演进数据飞轮，持续覆盖营销新业务场景。

- 【造数 Agent】基于 DAG + 马尔可夫状态机编排对话流程，[通用模型][用户模型]根据状态交互生成对话，多路径遍历长尾复杂场景。

- 【微调 Agent】融合【造数 Agent】新场景数据，微调 Qwen3-8B，构建[营销模型]新场景业务能力。

- 【对话 Agent】[营销模型] vs [用户模型]双模型对弈，注入用户画像&流转路径驱动多轮动态对话。

- 【质检 Agent】[质检模型] + 多规则并发 + 误判兜底，以人工标注为基准，评估[营销模型] P/R/F1。

- 【分析 Agent】Badcase 自动聚类 + 根因多维诊断，反向驱动【造数 Agent】定向生成样本。



## **DeepReview**

10月 2025

基于微调大模型与图谱建模的代码审查 SKILL，实现F1值(50% → 95%)、误判率(30% → 0%)、Token消耗(100% → 10%)，已发布 pip 包（deep-review）。

- 【数据】提取行内 16 个项目，共 12w+ 的 diff code，扩展至函数边界并注入状态和行号，高精度规则模型打标 + 低精度规则人工抽检构建训练集。

- 【微调】基于 Qwen3.8-27B 实施领域 SFT，提升《规则》校验精确率(45.56% → 94.76%)、召回率(84.08% → 80.32%)、F1值(57.64% → 86.94%)。

- 【图谱】基于 Tree-sitter AST 抽取实体（类/函数/文件/模块）与关系（调用/引用/继承/包含），通过 Leiden 算法完成高内聚社区划分与关键度评估。

- 【审查】以 diff 为中心进行 BFS 两跳链路追溯，剔除无关 code 构建最小精炼评审集，强制 Agent 评审前优先获取影响域上下文、禁止全盘扫描。



## **CosyVoice 推理延迟优化与分布式部署**

4月 2025

- 内容：分离式架构防止 LLM 和 token2wav 资源竞争；vLLM & 多 estimator 加速 LLM 推理速度；配套流式输入&输出，提升全流程速度。

- 成果：4 卡分布式部署下（4090），双流式 24 并发 P99 稳定在 429ms 内；消除流式输入延迟的条件下，TTS 推理 P99 低至 264ms，业界领先。



# 荣誉奖项
- 微众银行最佳拍档、协助之星、1024 Agent大赛二等奖（连续两届）；武汉研发中心双月之星、AI分享之星

- 国防科技大学全额学业奖学金（连续三届）、2023 优秀一等奖学金 & 优秀学员、2021 新生奖学金

- 郑州大学三级优秀毕业生（省级、校级、院级）、郑州大学一等奖学金 & 三好学生（连续四届）

- 2023 挑战杯学术科技作品赛（国二）、2022 大疆机甲大师挑战赛（国三）、2020 全国大学生信息安全竞赛（国三）

# 论文发表
## 领导者与协作者：解决多智能体强化学习中的稀疏奖励问题 | **Shaoqi Sun**, Hui Liu, Kele Xu

11月 2024

[10.1109/TETCI.2024.3488772](https://doi.org/10.1109/TETCI.2024.3488772) · IEEE Transactions on Artificial Intelligence



## 面向多智能体强化学习的双向影响与交互 | **Shaoqi Sun**, Kele Xu, Dawei Feng

5月 2024

[10.1109/TAI.2024.3401649](https://doi.org/10.1109/TAI.2024.3401649) · IEEE Transactions on Artificial Intelligence



## 基于时间不一致性的内在奖励在多智能体强化学习中的应用 | **Shaoqi Sun**, Kele Xu

8月 2023

[10.1109/IJCNN54540.2023.10191420](https://doi.org/10.1109/IJCNN54540.2023.10191420) · IJCNN CCF-C



## 渐进式多样化策略在多智能体强化学习中的应用 | **Shaoqi Sun**, Yuanzhao Zhai, Kele Xu

5月 2023

[10.1109/ICASSP49357.2023.10096125](https://doi.org/10.1109/ICASSP49357.2023.10096125) · ICASSP CCF-B



## 基于交互式多任务学习的中文文本情感分类 | Han Zhang（导师）, **Shaoqi Sun**, Yongjin Hu

7月 2020

[10.1109/ACCESS.2020.3007889](https://doi.org/10.1109/ACCESS.2020.3007889) · IEEE Access SCI



# 技术栈
**【ML & DL & RL】:** PLA、SVM、聚类、FNN、CNN、RNN、激活函数、优化器、DQN系列、AC系列、DDPG。

**【Transformer】:** 位置编码、ROPE、QKV、KV Cache、Attention、残差 & 归一化、MHA系列、Flash Attention。

**【SFT & RLHF】:** P-tuning、LoRA、QLoRA、蒸馏、PPO、GRPO、DRPO、GSPO、DPO、DAPO、VAPO。

**【vLLM & DeepSpeed】:** 显存估算、KV Cache、内存墙、PD分离、PagedAttention、并行策略、ZeRO(1-3)。

**【RAG】:** 混合检索、重排序、语义切分、Graph RAG、BGE、Qwen、BM25、IVF & HNSW、评估。

**【Agent】:** Qwen & DeepSeek、ReAct、MCP、COT、Skill、反思、多轮对话优化、长上下文处理。

**【Python & DA】:** 元组列表、面向对象、IO、多线程、NumPy、Pandas、Scikit-learn、Requests、Selenium。

**【分布式】:** 服务注册发现、CAP、阶段提交、负载均衡、限流熔断、共识算法、认证授权、Docker。

**【消息队列】:** 消费幂等性、可靠传输、顺序消费；解决消息挤压、耦合、削峰、分布式事务等问题。

**【缓存】:** 原理、数据结构、集群方式、内存碎片，RDB&AOF，分布式锁；缓存穿透（雪崩、击穿、污染）。

**【数据库】:** 数据库范式、事务处理、锁、SQL优化、InnoDB 引擎、MVCC 原理、索引机制。

**【数据结构】:** KMP、红黑树、堆、二叉树构造&遍历、BFS&DFS、最小生成树、最短路径、排序算法。

**【操作系统】:** 进程&线程、通信方式、用户态&核心态、内存管理、虚拟内存、Cache 替换算法。

**【计算机网络】:** TCP&UDP、TCP 流量控制、TCP 拥塞控制、三次握手&四次挥手、HTTP&HTTPS、DNS。

**【Java & JVM】:** 封装、继承、多态、多线程、集合、反射、内存模型、类加载、双亲委派、OOM、GC、JDK21。

**【Spring】:** IOC、AOP、Bean生命周期、循环依赖、事务传播机制；Spring MVC 拦截器&过滤器。
