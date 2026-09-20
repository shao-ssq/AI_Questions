// Import the rendercv function and all the refactored components
#import "@preview/rendercv:0.2.0": *

// Apply the rendercv template with custom configuration
#show: rendercv.with(
  name: "孙绍齐",
  title: "孙绍齐 - CV",
  footer: context { [#emph[孙绍齐 -- #str(here().page())\/#str(counter(page).final().first())]] },
  top-note: [ #emph[最后更新于 9月 2026] ],
  locale-catalog-language: "zh",
  text-direction: ltr,
  page-size: "us-letter",
  page-top-margin: 0.7in,
  page-bottom-margin: 0.7in,
  page-left-margin: 0.7in,
  page-right-margin: 0.7in,
  page-show-footer: true,
  page-show-top-note: true,
  colors-body: rgb(0, 0, 0),
  colors-name: rgb(0, 79, 144),
  colors-headline: rgb(0, 79, 144),
  colors-connections: rgb(0, 79, 144),
  colors-section-titles: rgb(0, 79, 144),
  colors-links: rgb(0, 79, 144),
  colors-footer: rgb(128, 128, 128),
  colors-top-note: rgb(128, 128, 128),
  typography-line-spacing: 0.6em,
  typography-alignment: "justified",
  typography-date-and-location-column-alignment: right,
  typography-font-family-body: "Source Sans 3",
  typography-font-family-name: "Source Sans 3",
  typography-font-family-headline: "Source Sans 3",
  typography-font-family-connections: "Source Sans 3",
  typography-font-family-section-titles: "Source Sans 3",
  typography-font-size-body: 10pt,
  typography-font-size-name: 20pt,
  typography-font-size-headline: 10pt,
  typography-font-size-connections: 10pt,
  typography-font-size-section-titles: 1.4em,
  typography-small-caps-name: false,
  typography-small-caps-headline: false,
  typography-small-caps-connections: false,
  typography-small-caps-section-titles: false,
  typography-bold-name: true,
  typography-bold-headline: false,
  typography-bold-connections: false,
  typography-bold-section-titles: true,
  links-underline: false,
  links-show-external-link-icon: false,
  header-alignment: center,
  header-photo-width: 3.5cm,
  header-space-below-name: 0.7cm,
  header-space-below-headline: 0.7cm,
  header-space-below-connections: 0.7cm,
  header-connections-hyperlink: true,
  header-connections-show-icons: true,
  header-connections-display-urls-instead-of-usernames: false,
  header-connections-separator: "",
  header-connections-space-between-connections: 0.5cm,
  section-titles-type: "with_partial_line",
  section-titles-line-thickness: 0.5pt,
  section-titles-space-above: 0.5cm,
  section-titles-space-below: 0.3cm,
  sections-allow-page-break: true,
  sections-space-between-text-based-entries: 0.4em,
  sections-space-between-regular-entries: 1.2em,
  entries-date-and-location-width: 2.3cm,
  entries-side-space: 0.2cm,
  entries-space-between-columns: 0.1cm,
  entries-allow-page-break: false,
  entries-short-second-row: true,
  entries-degree-width: 1cm,
  entries-summary-space-left: 0cm,
  entries-summary-space-above: 0cm,
  entries-highlights-bullet:  "•" ,
  entries-highlights-nested-bullet:  "•" ,
  entries-highlights-space-left: 0.15cm,
  entries-highlights-space-above: 0cm,
  entries-highlights-space-between-items: 0cm,
  entries-highlights-space-between-bullet-and-text: 0.5em,
  date: datetime(
    year: 2026,
    month: 9,
    day: 20,
  ),
)


= 孙绍齐

  #headline([算法工程师 | 软件工程硕士 | 1999.3(26) | 党员 | 男])

#connections(
  [#link("mailto:shaoqisun@qq.com", icon: false, if-underline: false, if-color: false)[#connection-with-icon("envelope")[shaoqisun\@qq.com]]],
  [#link("tel:+86-177-2483-8277", icon: false, if-underline: false, if-color: false)[#connection-with-icon("phone")[177 2483 8277]]],
  [#link("https://shao12138.blog.csdn.net/", icon: false, if-underline: false, if-color: false)[#connection-with-icon("link")[shao12138.blog.csdn.net]]],
)


== 教育背景

#education-entry(
  [
    #strong[国防科技大学（985，双一流高校）], 软件工程（A+，双一流学科）

    - 方向：多智能体强化学习（MARL）

    - 论文：基于策略差异化的多智能体内在奖励研究（IEEE Trans#sym.ast.basic#h(0pt, weak: true) 2、CCF-B、CCF-C）

    - 博客：CSDN 访问量 220w；LeetCode：716

  ],
  [
    湖南-长沙

    2021 – 2024

  ],
  degree-column: [
    #strong[硕士]
  ],
)

#education-entry(
  [
    #strong[郑州大学（211，双一流高校）], 软件工程

    - 方向：自然语言处理（NLP）

    - 论文：基于交互式多任务学习的中文文本情感分类（SCI）

    - 绩点：3.63\/4.00；排名：12\/712；英语：CET-6

  ],
  [
    河南-郑州

    2017 – 2021

  ],
  degree-column: [
    #strong[学士]
  ],
)

== 工作经历

#regular-entry(
  [
    #strong[微众银行 | 武汉研发中心 | 数据算法岗]

    - 方向：大语言模型微调（Fine-tune）、文本转语音算法（TTS）

    - 项目：AgentFlywheel（营销场景）、DeepReview、AOB 智能外呼系统（TTS推理加速）

    - 专利：MARL 下基于影响传播的奖励冲突解决方法、一种自适应波动流量下限流熔断策略

  ],
  [
    湖北-武汉

    2024 – 至今

  ],
)

== 项目经历

#regular-entry(
  [
    #strong[AgentFlywheel（微粒贷营销数据飞轮）]

    #summary[由造数、微调、对话、质检与分析五大 Agent 闭环联动，实现“造数→微调→对话→质检→分析→回流造数”的自演进数据飞轮，持续覆盖营销新业务场景。]

    - 【造数 Agent】基于 DAG + 马尔可夫状态机编排对话流程，\[通用模型\]\[用户模型\]根据状态交互生成对话，多路径遍历长尾复杂场景。

    - 【微调 Agent】融合【造数 Agent】新场景数据，微调 Qwen3-8B，构建\[营销模型\]新场景业务能力。

    - 【对话 Agent】\[营销模型\] vs \[用户模型\]双模型对弈，注入用户画像&流转路径驱动多轮动态对话。

    - 【质检 Agent】\[质检模型\] + 多规则并发 + 误判兜底，以人工标注为基准，评估\[营销模型\] P\/R\/F1。

    - 【分析 Agent】Badcase 自动聚类 + 根因多维诊断，反向驱动【造数 Agent】定向生成样本。

  ],
  [
    9月 2026

  ],
)

#regular-entry(
  [
    #strong[DeepReview]

    #summary[基于微调大模型与图谱建模的代码审查 SKILL，实现F1值(50\% → 95\%)、误判率(30\% → 0\%)、Token消耗(100\% → 10\%)，已发布 pip 包（deep-review）。]

    - 【数据】提取行内 16 个项目，共 12w+ 的 diff code，扩展至函数边界并注入状态和行号，高精度规则模型打标 + 低精度规则人工抽检构建训练集。

    - 【微调】基于 Qwen3.8-27B 实施领域 SFT，提升《规则》校验精确率(45.56\% → 94.76\%)、召回率(84.08\% → 80.32\%)、F1值(57.64\% → 86.94\%)。

    - 【图谱】基于 Tree-sitter AST 抽取实体（类\/函数\/文件\/模块）与关系（调用\/引用\/继承\/包含），通过 Leiden 算法完成高内聚社区划分与关键度评估。

    - 【审查】以 diff 为中心进行 BFS 两跳链路追溯，剔除无关 code 构建最小精炼评审集，强制 Agent 评审前优先获取影响域上下文、禁止全盘扫描。

  ],
  [
    10月 2025

  ],
)

#regular-entry(
  [
    #strong[CosyVoice 推理延迟优化与分布式部署]

    - 【内容】分离式架构防止 LLM 和 token2wav 资源竞争；vLLM & 多 estimator 加速 LLM 推理速度；配套流式输入&输出，提升全流程速度。

    - 【成果】4 卡分布式部署下（4090），双流式 24 并发 P99 稳定在 429ms 内；消除流式输入延迟的条件下，TTS 推理 P99 低至 264ms，业界领先。

  ],
  [
    4月 2025

  ],
)

== 荣誉奖项

- 微众银行最佳拍档、协助之星、1024 Agent大赛二等奖（连续两届）；武汉研发中心双月之星、AI分享之星

- 国防科技大学全额学业奖学金（连续三届）、2023 优秀一等奖学金 & 优秀学员、2021 新生奖学金

- 郑州大学三级优秀毕业生（省级、校级、院级）、郑州大学一等奖学金 & 三好学生（连续四届）

- 2023 挑战杯学术科技作品赛（国二）、2022 大疆机甲大师挑战赛（国三）、2020 全国大学生信息安全竞赛（国三）

== 论文发表

#regular-entry(
  [
    领导者与协作者：解决多智能体强化学习中的稀疏奖励问题 | #strong[Shaoqi Sun], Hui Liu, Kele Xu

    #link("https://doi.org/10.1109/TETCI.2024.3488772")[10.1109\/TETCI.2024.3488772] · IEEE Transactions on Artificial Intelligence

  ],
  [
    11月 2024

  ],
)

#regular-entry(
  [
    面向多智能体强化学习的双向影响与交互 | #strong[Shaoqi Sun], Kele Xu, Dawei Feng

    #link("https://doi.org/10.1109/TAI.2024.3401649")[10.1109\/TAI.2024.3401649] · IEEE Transactions on Artificial Intelligence

  ],
  [
    5月 2024

  ],
)

#regular-entry(
  [
    基于时间不一致性的内在奖励在多智能体强化学习中的应用 | #strong[Shaoqi Sun], Kele Xu

    #link("https://doi.org/10.1109/IJCNN54540.2023.10191420")[10.1109\/IJCNN54540.2023.10191420] · IJCNN CCF-C

  ],
  [
    8月 2023

  ],
)

#regular-entry(
  [
    渐进式多样化策略在多智能体强化学习中的应用 | #strong[Shaoqi Sun], Yuanzhao Zhai, Kele Xu

    #link("https://doi.org/10.1109/ICASSP49357.2023.10096125")[10.1109\/ICASSP49357.2023.10096125] · ICASSP CCF-B

  ],
  [
    5月 2023

  ],
)

#regular-entry(
  [
    基于交互式多任务学习的中文文本情感分类 | Han Zhang（导师）, #strong[Shaoqi Sun], Yongjin Hu

    #link("https://doi.org/10.1109/ACCESS.2020.3007889")[10.1109\/ACCESS.2020.3007889] · IEEE Access SCI

  ],
  [
    7月 2020

  ],
)

== 技术栈

#strong[【Agent】] LangChain（请求）、LangGraph（工作流）、DeepAgent（Agent）、ReAct（COT、反思）、MCP&Skill。

#strong[【SFT & RLHF】] P-tuning、LoRA、QLoRA、蒸馏、PPO、GRPO、DRPO、GSPO、DPO、DAPO、VAPO。

#strong[【vLLM & DeepSpeed】] 显存估算、KV Cache、内存墙、PD分离、PagedAttention、并行策略、ZeRO(1-3)。

#strong[【RAG】] 混合检索、重排序、语义切分、Graph RAG、BGE&Qwen3 Embedding、BM25&IVF、评估。

#strong[【LLM】] BERT、DeepSeek R1、Qwen系列、Prompt、复读、幻觉、漂移、长上下文（多轮对话）、评估。

#strong[【Transformer】] 位置编码、ROPE、QKV、KV Cache、Attention、残差 & 归一化、MHA系列、Flash Attention。

#strong[【NLP】] 分词、词嵌入、Word2vec、损失函数、语言模型LM（统计学、FNN、RNN）、seq2seq。

#strong[【强化学习】] 马尔可夫、贝尔曼方程、动态规划、时序拆分、DQN家族、AC家族、确定性策略梯度DDPG。

#strong[【深度学习】] 全连接网络FNN、循环神经网络RNN、卷积神经网络CNN、激活函数、优化器、梯度问题、学习率。

#strong[【机器学习】] 过拟合、损失函数、感知机算法、支持向量机、Rademacher、聚类、梯度下降、回归分类度量。

#strong[【数据分析】] 封装、继承、多态、多线程、元组列表、数据预处理、特征工程、数据挖掘、可视化分析。

#strong[【数据结构】] KMP、红黑树、堆、二叉树构造&遍历、BFS&DFS、最小生成树、最短路径、排序算法。

#strong[【操作系统】] 进程&线程、通信方式、用户态&核心态、内存管理、虚拟内存、Cache 替换算法。

#strong[【计算机网络】] TCP&UDP、TCP 流量控制、TCP 拥塞控制、三次握手&四次挥手、HTTP&HTTPS、DNS。

== 自我评价

- 求职意向：算法工程师。国防科技大学软件工程硕士（985\/双一流\/A+），现任微众银行数据算法岗（3年），深耕大模型微调、Agent 闭环架构与 AI 工程化落地利。

- 熟练掌握 LLM 领域微调与推理优化（vLLM、DeepSpeed、LoRA\/QLoRA），深度掌握 Agent 体系（LangGraph、DeepAgent、MCP\/Skill）与 RAG 建模，发表论文 5 篇，专利 2 项。

- 主导落地 AgentFlywheel 营销数据自演进飞轮；打造 DeepReview 代码审查工具，将审查 F1 值提升至 95\%、Token 消耗降低 90\%；优化 CosyVoice2\/3 语音推理使 P99 延迟低至 264ms，业界领先。
