# **大数据处理范式的演进：从批处理到流处理，再到批流一体的深度研究报告**

## **分布式计算的黎明：批处理时代的崛起与局限**

大数据处理的起点可以追溯到21世纪初，当时互联网的爆炸式增长使谷歌等公司面临着前所未有的海量数据挑战。在这一背景下，传统的单机存储和计算模型已无法支撑互联网规模的数据分析需求。2003年至2006年间，谷歌相继发表了三篇极具影响力的论文，分别论述了谷歌文件系统（GFS）、MapReduce编程模型以及BigTable分布式数据库，这三者后来被业界公认为大数据时代的“三驾马车” 1。

### **MapReduce的核心思想与核心论文**

MapReduce论文的核心思想在于通过“分而治之”的策略，将大规模计算任务拆解为在成百上千台廉价商品服务器上运行的并行任务 1。该模型巧妙地抽象了分布式系统的复杂性，包括数据分片、任务调度、容错处理和节点间通信，使开发者只需编写简单的Map（映射）和Reduce（归约）逻辑即可处理PB级的数据 2。

| 处理阶段 | 技术机制 | 数据流转特征 |
| :---- | :---- | :---- |
| 分片（Splitting） | 将大文件划分为固定大小的块（通常为64MB或128MB） | 数据局部性优先，尽量在存储数据的节点上计算 |
| 映射（Mapping） | 并行执行用户定义的Map函数，将输入记录转换为中间键值对 | 输出结果暂存在本地磁盘，不进入HDFS |
| 洗牌（Shuffling） | 根据键（Key）对中间结果进行分区、排序和合并 | 涉及大量的网络IO和磁盘读写，是性能瓶颈所在 |
| 归约（Reducing） | 将相同键的所有值聚合并输出最终结果 | 结果写回分布式文件系统，保证持久化 |

MapReduce极大地降低了大数据开发的门槛，但也确立了“离线处理即批处理”的初始范式。这种范式的核心在于“数据不动，代码动”，通过将计算逻辑推送到数据所在的节点，最大程度地减少了数据传输的开销 4。然而，MapReduce模型存在一个固有的缺陷：它是高度磁盘导向的。每一个处理阶段的结果都必须写入磁盘以保证容错性，这导致了极高的处理延迟，通常在数小时甚至数天之久 6。

### **离线处理的早期挑战与FlumeJava的出现**

随着谷歌内部MapReduce作业链的日益复杂，手动管理数十个相互依赖的任务变得异常困难 7。开发人员需要编写大量的协调代码来连接不同的MapReduce阶段，管理中间结果的清理，并处理各种故障场景 7。在这种背景下，谷歌开发了FlumeJava。

FlumeJava是一个基于Java的库，它引入了不可变的并行集合（如PCollection）和推迟评估（Deferred Evaluation）机制 7。开发者可以使用更高级的API定义数据流，而系统会自动构建执行计划图（DAG）并进行优化，例如将多个平行的Map操作融合（Fusion）进一个MapReduce作业中，从而显著提高执行效率并减少代码量 8。FlumeJava的成功不仅简化了谷歌内部的批处理流程，也为后来Apache Beam等框架的诞生奠定了理论基础 10。

## **实时需求的觉醒：流处理技术的演进**

进入2010年以后，随着社交媒体、移动支付和物联网的普及，企业对数据处理速度的要求从“天级”缩短到了“秒级” 11。在线广告竞价、金融欺诈监测以及实时推荐系统都要求数据一产生就能立即得到处理，这催生了实时流处理技术的爆发 12。

### **实时流处理的早期探索与MillWheel**

谷歌在2013年发表的MillWheel论文，系统地阐述了构建低延迟、高可靠流处理系统的挑战与方案 12。MillWheel的设计目标是实现互联网规模下的强一致性、容错性和低延迟 12。它引入了两个关键概念：持久化状态（Persistent State）和水位线（Low Watermarks） 12。

在流处理中，数据是持续不断的，系统必须记录中间状态（如当前窗口内的访问量），以便后续计算 12。MillWheel通过在每个处理节点维护一个与主逻辑一致的状态库，并结合精确一次（Exactly-once）的处理语义，确保了即使在节点故障时，计算结果也不会出现偏差 14。水位线则通过追踪数据的产生时间（Event Time）而非处理时间（Processing Time），为乱序到达的数据提供了处理依据 12。

| 处理范式 | 处理单元 | 延迟特征 | 典型应用 |
| :---- | :---- | :---- | :---- |
| 批处理 | 静态、完整的数据集 | 小时/天 | 历史报告、深度挖掘、离线训练 |
| 流处理 | 动态、无穷的数据流 | 毫秒/秒 | 实时看板、安全预警、个性化推荐 |

## **Lambda架构：应对现实的妥协之作**

尽管流处理技术在不断进步，但在21世纪10年代初期，流处理框架（如Apache Storm）在处理大规模历史数据时的吞吐量和准确性仍不及成熟的批处理引擎 17。为了同时满足实时性和准确性的双重需求，Nathan Marz提出了著名的Lambda架构 17。

### **Lambda架构的分层思想**

Lambda架构的核心思想是将数据处理流程拆分为三个相互补充的层 19：

1. **批处理层（Batch Layer）**：负责管理主数据集，并在一个固定的时间窗口（如每天）重新计算所有的历史数据，生成精确的批处理视图 19。  
2. **加速层（Speed Layer）**：负责实时处理新到达的数据，提供低延迟的数据视图。由于它只处理增量数据，且不具备全局视野，其结果可能是近似的 19。  
3. **服务层（Serving Layer）**：将批处理视图和实时视图合并，响应用户的查询请求 19。

### **Lambda架构的痛点：双重复杂性**

Lambda架构在很长一段时间内成为了大数据领域的标准实践，但它也带来了巨大的运维压力 17。最突出的问题是代码重复：开发人员必须用两种不同的技术栈实现同一套业务逻辑（例如用MapReduce写批处理代码，用Storm写实时代码） 17。这种双重实现不仅增加了开发成本，还容易导致“逻辑漂移”，即批处理层和加速层对同一份数据的处理逻辑在细节上出现偏差，导致查询结果不一致 21。

## **Kappa架构：迈向统一的尝试**

为了解决Lambda架构的复杂性问题，LinkedIn的Jay Kreps在2014年提出了Kappa架构 22。其核心论点是：如果流处理引擎能够处理足够大的吞吐量，并且能够通过重放历史日志来进行重新计算，那么批处理层就是多余的 22。

### **Kappa架构的逻辑与机制**

Kappa架构建议将所有数据都视为流，并将这些流持久化在类似Apache Kafka这样的分布式消息队列中 22。当业务逻辑发生变更需要重新计算时，Kappa架构不再依赖另一个批处理系统，而是启动一个新的流处理作业实例，从Kafka的起始偏移量（Offset）开始重放历史数据 22。

| 操作步骤 | Kappa架构重新计算流程 | 预期目标 |
| :---- | :---- | :---- |
| 第一步 | 启动流处理作业的新版本代码实例 | 引入新的业务逻辑 |
| 第二步 | 从数据存储系统（如Kafka）的起始位置重放数据 | 覆盖历史数据区间 |
| 第三步 | 将结果写入新的输出表中，并与旧结果进行比对 | 验证逻辑准确性 |
| 第四步 | 切换下游应用到新输出表，并清理旧任务 | 完成系统升级 |

Kappa架构极大地简化了系统设计，使得企业只需维护一套代码和一套处理框架 22。然而，它对底层存储和流处理引擎提出了极高的要求：存储系统必须能够保存极长时间的原始数据，且流处理引擎在进行历史追溯时必须具备极高的吞吐性能 22。

## **Dataflow模型：批流一体的哲学根基**

在Lambda和Kappa架构争鸣的同时，谷歌内部正在进行一场更深层次的理论革命。2015年，Tyler Akidau等人发表了《Dataflow模型：在大规模、无界、无序数据处理中平衡准确性、延迟与成本的实用方法》一文 16。这篇论文不仅奠定了谷歌云Dataflow服务的基础，更成为了整个大数据行业实现“批流一体”的哲学指南 10。

### **核心见解：打破“批”与“流”的虚假对立**

Dataflow模型认为，不应使用“批”和“流”来描述数据处理的方式，而应关注数据本身是“有界”（Bounded）还是“无界”（Unbounded）的 27。批处理只是有界数据处理的一种特例，而流处理则是无界数据处理的常态 28。

该模型提出了处理无界数据的四个关键问题（The Four W's） 16：

1. **What（计算什么内容）**：通过变换（Transformations）定义具体的计算逻辑，如Sum或Count 26。  
2. **Where（在哪个时间区间计算）**：通过窗口（Windowing）将无界数据切分为有限的区间，支持固定窗口、滑动窗口和会话窗口 16。  
3. **When（何时触发输出结果）**：通过触发器（Triggers）决定在处理时间的哪个点物化计算结果。这允许系统根据水位线或记录数灵活决定输出时机 16。  
4. **How（结果如何修正）**：通过累积模式（Refinement Modes）定义新到的数据如何影响已输出的结果，包括丢弃、累积和累积撤回等模式 16。

### **时间语义的精确区分**

Dataflow模型强调了\*\*事件时间（Event Time）**和**处理时间（Processing Time）\*\*的根本区别 16。在现实场景中，两者之间总是存在偏移（Skew），这是由于网络延迟、硬件故障或系统背压造成的 26。Dataflow通过水位线（Watermarks）机制，赋予了开发者在不可预测的延迟下依然能够获得准确计算结果的能力 26。

## **Apache Flink：批流一体的工程化巅峰**

如果说Dataflow模型提供了理论指导，那么Apache Flink则是将这些理论付诸实践的最成功范例 28。Flink的诞生理念正是“批处理是流处理的一个特例”，这使得它从架构底层就天然具备了批流一体的基因 18。

### **阿里与Blink：实战中的进化**

阿里巴巴是Flink在全球范围内最重要的推动者之一。2015年，阿里搜索推荐团队在调研后认为，传统的Storm无法满足双11对超大规模数据的一致性处理需求，而Spark Streaming的微批处理架构在极低延迟场景下表现欠佳 3。于是，阿里选择了Flink，并根据自身业务需求进行了深度定制，这个内部版本被称为Blink 18。

Blink在阿里内部的演进历程中解决了多个关键挑战：

1. **大规模状态管理**：双11期间产生的状态数据达到PB级别。Blink优化了RocksDB状态存储后端，引入了增量快照和本地磁盘优化，显著降低了检查点（Checkpoint）对性能的影响 3。  
2. **SQL层的统一**：阿里团队在Flink上投入了大量资源开发SQL接口，使得业务方可以通过标准的SQL语句同时定义实时和离线任务，实现了“一套代码，两种运行模式”的批流一体愿景 18。  
3. **调度与容错的增强**：针对离线处理的特点，Blink引入了插拔式的调度器和细粒度的故障恢复机制。在输入数据有界时，系统能够像传统批处理器一样进行分阶段调度和数据落盘，极大提升了资源利用率 18。

| 时间点 | 阿里巴巴Flink/Blink关键里程碑 | 业务规模与成效 |
| :---- | :---- | :---- |
| 2015年 | 阿里开始调研流处理引擎，选择Flink作为核心方向 | 内部搜索、推荐场景试点 |
| 2016年 | 首次支持双11，处理搜索与推荐索引构建 | 实时计算平台正式上线 |
| 2017年 | 阿里核心业务（如GMV大屏）全面迁移至Blink | 峰值处理能力提升1倍以上 |
| 2019年 | 阿里巴巴宣布将Blink源码贡献给Apache Flink社区 | Flink 1.9版本引入大量Blink特性 |
| 2020年 | 实现全链路批流一体化处理核心业务数据 | 峰值处理速度达40亿条/秒，数据量7TB/秒 |

### **核心技术：Chandy-Lamport算法与状态一致性**

Flink之所以能实现高效的批流一体，离不开其对Chandy-Lamport分布式快照算法的改进 3。通过在数据流中插入特殊的Barrier（屏障），Flink能够在不停止处理的前提下，周期性地触发分布式状态的一致性持久化 3。对于批处理任务，Flink可以禁用这些Barrier，转而采用阻塞式洗牌（Blocking Shuffle）和分阶段重试机制，从而获得与传统离线引擎相当的稳定性 18。

## **批流一体的高级形态：流式湖仓（Streaming Lakehouse）**

随着批流一体在计算引擎层面逐步成熟，行业开始意识到存储层面的隔离依然是实现最终目标的阻碍 36。传统架构中，实时数据流向Kafka，历史数据流向HDFS/S3，这种物理上的分离迫使企业仍需维护两套存储系统 37。这促使了“流式湖仓”概念的兴起。

### **统一存储层：Apache Paimon与Iceberg**

以Apache Paimon、Apache Iceberg和Delta Lake为代表的新一代表格式（Table Formats），正试图统一有界和无界数据的存储逻辑 32。

1. **Apache Paimon**：作为Flink生态中原生的流式湖仓项目，Paimon采用了LSM-tree存储结构，能够支持每秒数百万次的增量写入和实时读取 32。它允许Flink直接在对象存储上进行分钟甚至秒级的增量物化，使得“离线表”和“实时流”在存储层合二为一 38。  
2. **Lakehouse 2.0的特征**：现代湖仓架构不仅支持ACID事务，还支持增量合并（Upsert）和时间旅行（Time Travel）查询 36。这意味着，同一个数据表既可以作为Kafka一样的消息源被流处理引擎实时订阅，也可以作为Hive表接受离线SQL的深度扫描 36。

| 存储特性 | 传统数据湖 (Hive) | 实时消息系统 (Kafka) | 流式湖仓 (Paimon/Iceberg) |
| :---- | :---- | :---- | :---- |
| 数据可见性 | 小时/天 (T+1) | 毫秒级 | 分钟/秒级 |
| 存储成本 | 极低 (对象存储/HDFS) | 高 (昂贵的磁盘/有限留存) | 低 (存放在对象存储，支持海量留存) |
| 查询能力 | 强 (全表扫描/索引) | 弱 (仅支持点对点消费) | 强 (支持OLAP加速与流式读取) |
| 事务支持 | 无 | 单条消息级别 | 完整ACID支持 |

### **数据新鲜度的阶跃提升**

在湖仓一体的加持下，数据的流转效率得到了质的提升。以阿里的典型链路为例，数据通过Flink写入Paimon，下游任务可以立刻看到新产生的变更日志（Changelog），而无需等待传统的批处理调度任务 38。这种架构不仅消除了Lambda架构中的层级累积延迟，还通过减少数据冗余，将存储成本降低了40%以上 32。

## **计算与开发的标准化：Apache Beam的桥梁作用**

在大数据技术快速演进的过程中，多样化的引擎和API给企业带来了严重的“供应商锁定”风险 40。Apache Beam作为一种统一的编程模型，旨在解决这一问题。

### **编程模型的抽象化**

Beam的理念是“一次编写，到处运行” 40。它将数据处理逻辑抽象为PCollection和PTransform，开发者只需关注逻辑本身，而不必关心底层执行引擎是Flink、Spark还是Google Cloud Dataflow 42。这种抽象不仅提高了代码的可移植性，还促使了跨语言处理（Cross-language pipelines）的实现，允许在同一个流水线中混合使用Java、Python和Go语言编写的任务 40。

### **统一SQL的愿景**

为了进一步降低非技术人员的使用门槛，Beam和Flink都投入了大量的精力在SQL层面上。通过引入Apache Calcite作为解析器和优化器，这些框架能够将复杂的SQL查询转换为高性能的分布式执行计划 34。这种“SQL即代码”的模式，真正实现了让业务分析师能够直接参与实时与离线数据分析的“民主化”目标 34。

## **行业案例分析：从理论到百亿级实战**

### **阿里巴巴双11的巅峰考验**

阿里的批流一体实践是大数据史上最具代表性的案例之一。在2020年的双11期间，阿里的实时计算平台不仅支撑了媒体大屏的GMV展示，更深入到了核心的业务决策链路中 45。

1. **全链路一致性**：通过Flink的批流一体架构，阿里实现了实时报表与离线报表的完全一致。在此之前，运营人员经常发现实时看板的数字与第二天跑出的离线统计数字对不上，这在很大程度上影响了决策的及时性 46。  
2. **资源的高效复用**：双11当天的零点是实时流量的峰值，而此时离线计算资源相对闲置。批流一体化使得阿里能够在实时流量高峰期将计算资源优先分配给实时任务，而在凌晨流量回落后迅速切换资源执行离线清理和深度分析，极大地提升了整体硬件的性价比 46。

### **领英（LinkedIn）与Apache Beam的规模化**

LinkedIn作为Kappa架构的发源地，目前每天通过3000多条Apache Beam流水线处理超过4万亿条事件 40。通过统一的Beam模型，LinkedIn不仅实现了实时与离线逻辑的共享，还将其基础设施成本降低了约50% 40。这证明了即使在百亿级规模下，统一模型依然具备极高的执行效率和经济效益。

## **深度洞察：演进背后的逻辑与社会技术因素**

大数据处理范式的演进并非单纯的技术驱动，而是由业务需求、硬件进步以及工程哲学共同催化的结果。

### **硬件革新的助推作用**

存储技术的演进对批流一体的实现至关重要。非易失性存储器（NVM）、高速SSD以及RDMA网络技术的应用，打破了传统磁盘IO对实时处理的束缚 47。在MapReduce时代，数据交换必须落盘是因为内存昂贵且不可靠；而在Flink时代，我们可以通过精细的状态管理将大部分热数据保存在内存中，利用现代硬件的吞吐能力实现低延迟的强一致性 33。

### **组织结构的变革**

批流一体不仅改变了代码，还改变了数据团队的组织方式。在Lambda架构时代，公司通常需要分别维护一个“离线团队”和一个“实时团队”，这两个团队之间往往存在严重的沟通壁垒 17。批流一体的成熟，使得企业可以构建统一的“数据工程团队”，专注于定义数据流的业务价值，而不是纠结于技术栈的选择。

## **结论：大数据的终极形态？**

从2004年的MapReduce到2024年的流式湖仓，大数据处理完成了从“静态批处理”到“动态流处理”，再到“统一流计算”的完整蜕变。

### **演进逻辑总结**

1. **阶段一：混沌中的秩序（2004-2010）**。MapReduce解决了“能算”的问题，通过牺牲延迟换取了前所未有的处理规模 4。  
2. **阶段二：速度的追求（2011-2014）**。流处理框架层出不穷，但Lambda架构的复杂性揭示了流与批之间的深刻裂痕 17。  
3. **阶段三：理论的统一（2015-2018）**。Dataflow模型和Flink的崛起，从语义和工程上证明了“批是流的特例”，奠定了批流一体的基石 16。  
4. **阶段四：存储的闭环（2019至今）**。流式湖仓的成熟，标志着批流一体从计算引擎延伸到了存储底座，大数据系统正朝着更简单、更实时、更智能的方向迈进 36。

未来的大数据处理可能将不再有“离线”和“实时”之分。随着人工智能与大模型的深度融合，数据处理系统将演变为一种“持续智能引擎”，它像生物的神经系统一样，不断地感知、学习并对现实世界做出即时反应 36。对于企业而言，批流一体不再是一个可选的技术优化，而是实现数字化转型的核心基础设施。

在这种背景下，开发者应当关注的是如何利用水位线处理时间上的不确定性，如何通过状态机处理业务逻辑的连贯性，以及如何通过湖仓统一架构降低系统的熵增。大数据演进的故事告诉我们：技术的终极目标总是回归简单，而这种简单是建立在对复杂性深刻理解的基础之上的。

#### **引用的著作**

1. What is MapReduce? \- IBM, 访问时间为 二月 3, 2026， [https://www.ibm.com/think/topics/mapreduce](https://www.ibm.com/think/topics/mapreduce)  
2. MapReduce: Simplified Data Processing on Large Clusters \- Google Research, 访问时间为 二月 3, 2026， [https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/?ref=io-net.ghost.io](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/?ref=io-net.ghost.io)  
3. Why Did Alibaba Choose Apache Flink Anyway? \- Alibaba Cloud Community, 访问时间为 二月 3, 2026， [https://www.alibabacloud.com/blog/why-did-alibaba-choose-apache-flink-anyway\_595190](https://www.alibabacloud.com/blog/why-did-alibaba-choose-apache-flink-anyway_595190)  
4. Demystifying Distributed Systems: An Overview of Google's ..., 访问时间为 二月 3, 2026， [https://medium.com/@duhov/demystifying-distributed-systems-an-overview-of-googles-mapreduce-paper-00b664481fc3](https://medium.com/@duhov/demystifying-distributed-systems-an-overview-of-googles-mapreduce-paper-00b664481fc3)  
5. MapReduce \- Wikipedia, 访问时间为 二月 3, 2026， [https://en.wikipedia.org/wiki/MapReduce](https://en.wikipedia.org/wiki/MapReduce)  
6. Batch vs. streaming data processing in Databricks, 访问时间为 二月 3, 2026， [https://docs.databricks.com/aws/en/data-engineering/batch-vs-streaming](https://docs.databricks.com/aws/en/data-engineering/batch-vs-streaming)  
7. FlumeJava: Easy, Efficient Data-Parallel Pipelines \- Google Research, 访问时间为 二月 3, 2026， [https://research.google.com/pubs/archive/35650.pdf](https://research.google.com/pubs/archive/35650.pdf)  
8. reading-notes/papers/010\_FlumeJava\_Easy\_Efficient\_Data-Parallel\_Pipelines.md at master \- GitHub, 访问时间为 二月 3, 2026， [https://github.com/florian/reading-notes/blob/master/papers/010\_FlumeJava\_Easy\_Efficient\_Data-Parallel\_Pipelines.md](https://github.com/florian/reading-notes/blob/master/papers/010_FlumeJava_Easy_Efficient_Data-Parallel_Pipelines.md)  
9. FlumeJava: Easy, Efficient Data-Parallel Pipelines \- Google Research, 访问时间为 二月 3, 2026， [https://research.google/pubs/flumejava-easy-efficient-data-parallel-pipelines/](https://research.google/pubs/flumejava-easy-efficient-data-parallel-pipelines/)  
10. The Past and Present of Stream Processing (Part 12): The Cost of Abstraction — Apache Beam, Google's Legacy \- Gang Tao, 访问时间为 二月 3, 2026， [https://taogang.medium.com/the-past-and-present-of-stream-processing-part-12-the-cost-of-abstraction-apache-beam-d50fc7baac90](https://taogang.medium.com/the-past-and-present-of-stream-processing-part-12-the-cost-of-abstraction-apache-beam-d50fc7baac90)  
11. Real-Time Data Architecture: Moving Beyond Batch Processing \- Systech Solutions, 访问时间为 二月 3, 2026， [https://systechusa.com/realtime-data-architecture/](https://systechusa.com/realtime-data-architecture/)  
12. MillWheel: Fault-Tolerant Stream Processing at Internet Scale \- Google Research, 访问时间为 二月 3, 2026， [https://research.google.com/pubs/archive/41378.pdf](https://research.google.com/pubs/archive/41378.pdf)  
13. A Flink Series from the Alibaba Tech Team \- Medium, 访问时间为 二月 3, 2026， [https://alibabatech.medium.com/a-flink-series-from-the-alibaba-tech-team-b8b5539fdc70](https://alibabatech.medium.com/a-flink-series-from-the-alibaba-tech-team-b8b5539fdc70)  
14. Insights from Paper: Google MillWheel: Fault-Tolerant Stream Processing at Internet Scale, 访问时间为 二月 3, 2026， [https://hemantkgupta.medium.com/insights-from-paper-google-millwheel-fault-tolerant-stream-processing-at-internet-scale-4d8fb7686391](https://hemantkgupta.medium.com/insights-from-paper-google-millwheel-fault-tolerant-stream-processing-at-internet-scale-4d8fb7686391)  
15. MillWheel: Fault-Tolerant Stream Processing at Internet Scale (2013) \- Papers, 访问时间为 二月 3, 2026， [https://mwhittaker.github.io/papers/html/akidau2013millwheel.html](https://mwhittaker.github.io/papers/html/akidau2013millwheel.html)  
16. The Dataflow Model: A Practical Approach to Balancing Correctness ..., 访问时间为 二月 3, 2026， [https://blog.acolyer.org/2015/08/18/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/](https://blog.acolyer.org/2015/08/18/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/)  
17. Revisiting the Lambda Architecture: Challenges and Alternatives | by CortexFlow | The Software Frontier | Medium, 访问时间为 二月 3, 2026， [https://medium.com/the-software-frontier/revisiting-the-lambda-architecture-challenges-and-alternatives-1da4fd5ce08f](https://medium.com/the-software-frontier/revisiting-the-lambda-architecture-challenges-and-alternatives-1da4fd5ce08f)  
18. Batch as a Special Case of Streaming and Alibaba's contribution of Blink | Apache Flink, 访问时间为 二月 3, 2026， [https://flink.apache.org/2019/02/13/batch-as-a-special-case-of-streaming-and-alibabas-contribution-of-blink/](https://flink.apache.org/2019/02/13/batch-as-a-special-case-of-streaming-and-alibabas-contribution-of-blink/)  
19. What is Lambda Architecture? | Dremio, 访问时间为 二月 3, 2026， [https://www.dremio.com/wiki/lambda-architecture/](https://www.dremio.com/wiki/lambda-architecture/)  
20. What Is Lambda Architecture? | Scalable Data Processing \- Actian Corporation, 访问时间为 二月 3, 2026， [https://www.actian.com/glossary/lambda-architecture/](https://www.actian.com/glossary/lambda-architecture/)  
21. Revisiting the Lambda Architecture: Challenges and Alternatives \- DEV Community, 访问时间为 二月 3, 2026， [https://dev.to/cortexflow/revisiting-the-lambda-architecture-challenges-and-alternatives-1koa](https://dev.to/cortexflow/revisiting-the-lambda-architecture-challenges-and-alternatives-1koa)  
22. Kappa Architecture in Action — From Sensors to Dashboards with ..., 访问时间为 二月 3, 2026， [https://blog.dataengineerthings.org/kappa-architecture-in-action-from-sensors-to-dashboards-with-kafka-spark-streaming-starrocks-80492a509e1a](https://blog.dataengineerthings.org/kappa-architecture-in-action-from-sensors-to-dashboards-with-kafka-spark-streaming-starrocks-80492a509e1a)  
23. Kappa Architecture \- Where Every Thing Is A Stream \- Milinda Pathirage, 访问时间为 二月 3, 2026， [https://milinda.pathirage.org/kappa-architecture.com/](https://milinda.pathirage.org/kappa-architecture.com/)  
24. Data processing architectures – Lambda and Kappa \- Ericsson, 访问时间为 二月 3, 2026， [https://www.ericsson.com/en/blog/2015/11/data-processing-architectures--lambda-and-kappa](https://www.ericsson.com/en/blog/2015/11/data-processing-architectures--lambda-and-kappa)  
25. The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing \- Google Research, 访问时间为 二月 3, 2026， [https://research.google.com/pubs/archive/43864.pdf](https://research.google.com/pubs/archive/43864.pdf)  
26. The Stream Processing Model Behind Google Cloud Dataflow | Towards Data Science, 访问时间为 二月 3, 2026， [https://towardsdatascience.com/the-stream-processing-model-behind-google-cloud-dataflow-0d927c9506a0/](https://towardsdatascience.com/the-stream-processing-model-behind-google-cloud-dataflow-0d927c9506a0/)  
27. The Dataflow Model. A paper review | by Heather Arthur | Jan, 2026 \- Medium, 访问时间为 二月 3, 2026， [https://medium.com/@heathermoor/the-dataflow-model-520f486a56bf](https://medium.com/@heathermoor/the-dataflow-model-520f486a56bf)  
28. What is Apache Flink? \- AWS, 访问时间为 二月 3, 2026， [https://aws.amazon.com/what-is/apache-flink/](https://aws.amazon.com/what-is/apache-flink/)  
29. Batch and Stream Processing in Confluent Cloud for Apache Flink, 访问时间为 二月 3, 2026， [https://docs.confluent.io/cloud/current/flink/concepts/batch-and-stream-processing.html](https://docs.confluent.io/cloud/current/flink/concepts/batch-and-stream-processing.html)  
30. Streaming pipelines | Cloud Dataflow \- Google Cloud Documentation, 访问时间为 二月 3, 2026， [https://docs.cloud.google.com/dataflow/docs/concepts/streaming-pipelines](https://docs.cloud.google.com/dataflow/docs/concepts/streaming-pipelines)  
31. Watermarks in Stream Processing Systems: Semantics and Comparative Analysis of Apache Flink and Google Cloud Dataflow \- VLDB Endowment, 访问时间为 二月 3, 2026， [http://www.vldb.org/pvldb/vol14/p3135-begoli.pdf](http://www.vldb.org/pvldb/vol14/p3135-begoli.pdf)  
32. Introduction to Unified Batch and Stream Processing of Apache Flink ..., 访问时间为 二月 3, 2026， [https://www.alibabacloud.com/blog/introduction-to-unified-batch-and-stream-processing-of-apache-flink\_601407](https://www.alibabacloud.com/blog/introduction-to-unified-batch-and-stream-processing-of-apache-flink_601407)  
33. Fight Peak Data Traffic On 11.11: The Secrets of Alibaba Stream Computing, 访问时间为 二月 3, 2026， [https://102.alibaba.com/detail?id=35](https://102.alibaba.com/detail?id=35)  
34. Stream Processing for Everyone with SQL and Apache Flink, 访问时间为 二月 3, 2026， [https://flink.apache.org/2016/05/24/stream-processing-for-everyone-with-sql-and-apache-flink/](https://flink.apache.org/2016/05/24/stream-processing-for-everyone-with-sql-and-apache-flink/)  
35. Apache Flink \- Wikipedia, 访问时间为 二月 3, 2026， [https://en.wikipedia.org/wiki/Apache\_Flink](https://en.wikipedia.org/wiki/Apache_Flink)  
36. The Lakehouse 2.0 Era: Merging Streaming and Batch in a Unified ..., 访问时间为 二月 3, 2026， [https://medium.com/@manik.ruet08/the-lakehouse-2-0-era-merging-streaming-and-batch-in-a-unified-model-c1e46cedcf47](https://medium.com/@manik.ruet08/the-lakehouse-2-0-era-merging-streaming-and-batch-in-a-unified-model-c1e46cedcf47)  
37. Lakehouse Architecture for Unified Batch and Streaming Data Processing Marvellous Desell, 访问时间为 二月 3, 2026， [https://www.researchgate.net/publication/398498927\_Lakehouse\_Architecture\_for\_Unified\_Batch\_and\_Streaming\_Data\_Processing\_Marvellous\_Desell](https://www.researchgate.net/publication/398498927_Lakehouse_Architecture_for_Unified_Batch_and_Streaming_Data_Processing_Marvellous_Desell)  
38. Towards A Unified Streaming & Lakehouse Architecture | Apache Fluss™ (Incubating), 访问时间为 二月 3, 2026， [https://fluss.apache.org/blog/unified-streaming-lakehouse/](https://fluss.apache.org/blog/unified-streaming-lakehouse/)  
39. The Past, Present, and Future of Apache Flink \- Alibaba Cloud Community, 访问时间为 二月 3, 2026， [https://www.alibabacloud.com/blog/the-past-present-and-future-of-apache-flink\_601867](https://www.alibabacloud.com/blog/the-past-present-and-future-of-apache-flink_601867)  
40. Apache Beam®, 访问时间为 二月 3, 2026， [https://beam.apache.org/](https://beam.apache.org/)  
41. Apache Beam: Introduction to Batch and Stream Data Processing \- Confluent, 访问时间为 二月 3, 2026， [https://www.confluent.io/learn/apache-beam/](https://www.confluent.io/learn/apache-beam/)  
42. Apache Beam vs Apache Flink: Which One Suits You Best? \- RisingWave, 访问时间为 二月 3, 2026， [https://risingwave.com/blog/apache-beam-vs-apache-flink-which-one-suits-you-best/](https://risingwave.com/blog/apache-beam-vs-apache-flink-which-one-suits-you-best/)  
43. Apache Beam vs Flink: The Definitive Guide to Stream Processing \- Wallarm, 访问时间为 二月 3, 2026， [https://www.wallarm.com/cloud-native-products-101/apache-beam-vs-flink-stream-processing](https://www.wallarm.com/cloud-native-products-101/apache-beam-vs-flink-stream-processing)  
44. Apache Beam: How Beam Runs on Top of Flink, 访问时间为 二月 3, 2026， [https://flink.apache.org/2020/02/22/apache-beam-how-beam-runs-on-top-of-flink/](https://flink.apache.org/2020/02/22/apache-beam-how-beam-runs-on-top-of-flink/)  
45. Four Billion Records per Second\! What is Behind Alibaba Double 11 \- Flink Stream-Batch Unification Practice during Double 11 for the Very First Time, 访问时间为 二月 3, 2026， [https://www.alibabacloud.com/blog/four-billion-records-per-second-what-is-behind-alibaba-double-11---flink-stream-batch-unification-practice-during-double-11-for-the-very-first-time\_596962](https://www.alibabacloud.com/blog/four-billion-records-per-second-what-is-behind-alibaba-double-11---flink-stream-batch-unification-practice-during-double-11-for-the-very-first-time_596962)  
46. Apache Flink's stream-batch unification powers Alibaba's 11.11 in 2020 \- Ververica, 访问时间为 二月 3, 2026， [https://www.ververica.com/blog/apache-flinks-stream-batch-unification-powers-alibabas-11.11-in-2020](https://www.ververica.com/blog/apache-flinks-stream-batch-unification-powers-alibabas-11.11-in-2020)  
47. Facing the Technical Challenges of Double 11 \- YouTube, 访问时间为 二月 3, 2026， [https://www.youtube.com/watch?v=ii-oj3\_ggaA](https://www.youtube.com/watch?v=ii-oj3_ggaA)  
48. Architecture | Apache Flink, 访问时间为 二月 3, 2026， [https://flink.apache.org/what-is-flink/flink-architecture/](https://flink.apache.org/what-is-flink/flink-architecture/)  
49. Unified batch and stream processing architecture design patterns \- ResearchGate, 访问时间为 二月 3, 2026， [https://www.researchgate.net/publication/392073388\_Unified\_batch\_and\_stream\_processing\_architecture\_design\_patterns](https://www.researchgate.net/publication/392073388_Unified_batch_and_stream_processing_architecture_design_patterns)