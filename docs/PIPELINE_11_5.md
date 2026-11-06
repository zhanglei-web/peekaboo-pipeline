# 11-5 直接帧与双接口流水线

日期：2026-11-05。代码在仓库`/home/baai/zl/todo/11.4/processing`的`11-5`分支。`11-4`标签、旧脚本与历史数据均保留。新入口`pipeline115.sh`和包`pipeline115/`独立于旧入口。

后续剩余72条的输入清单、位置事实附加字段及用户批准的右目单帧缺项，见[本批运行文档](BATCH_REMAINING72_20261105.md)。下方单条验收记录对应`11-5-dev`里程碑，不能作为72条已完成的证明。

## 1. 已实现范围与部署决定

新流程为输入内容审计 → CPU VIO/运动筛帧/校正 → GPU深度 → 直接帧包完整校验 → Astra完整episode粗筛/精筛 → Qwen场景 → 全部事件描述 → 按顺序更新时序树 → 组装/验证/两目标格式导出。没有rosbag写出、读回或提取阶段。每条完成深度后即可进入Astra，无整批29条屏障。

当前接口已定位：

|接口|模型与部署|当前阶段分工|
|---|---|---|
|Astra|沿用固定的`5.demo.json`与`ZYAPI_API_KEY`|场景粗筛、精筛|
|本机Qwen|127.0.0.1:8080/v1，local-vlm，Q4_K_M，16384上下文|场景、时序更新；与深度GPU分时|
|局域网Qwen|192.168.99.99:8089/v1，qwen3.8-27b，鉴权模型清单确认Q8_0、8192上下文|事件描述，密钥环境`QWEN_API_KEY`|

用户最新确认服务器地址改为`192.168.99.99`，沿用8089端口实际health返回200/ok；使用用户提供的本机密钥文件鉴权后，`/v1/models`返回200，确认模型名`qwen3.8-27b`、Q8_0及n_ctx=8192。密钥仅传入子进程环境，不进入配置、日志或Git。原`192.168.20.108`的历史业务测试只作参考，不能替代新地址的新推理验收。

原计划的“一条episode全程交给任一Qwen worker”作了明确调整：LAN单槽8192上下文小于原场景输出预算10000，不能无损承接场景。故以场景/事件/时序阶段为交接边界，本机负责场景和时序，LAN负责事件。不同episode可在两个接口上重叠；同一episode的时序更新严格顺序、只有一个写者。更改`routes`可显式指定分工；不会失败后自动切换端点。

场景10000、事件6144、时序3072的输出预算，temperature0、seed42、关闭思考，以及原提示词/schema均保留。客户端先排除输出预算本身已超过上下文的配置；包含图像、schema和模板后的完整输入以服务端实际容量校验为准。LAN事件若报上下文不足，记录错误进入needs_repair，不降低预算、不缩图、不删字段。

两个量化与视觉投影不同，不能承诺逐字相同或语义质量相等。历史LAN事件测试有结构通过但过度推断和翻译目标名的问题；本版不将其视为完整质量验收。模型身份、量化声明、接口、提示词、图像哈希、schema、预算和响应都分开留证。远端权重哈希未独立核查。

## 2. 终端运行

本机已安装依赖，包装脚本自动加载ROS Jazzy。Python依赖、模型、OpenVINS和深度权重沿用11-4，150项冻结代码哈希仍需匹配。

```bash
cd /home/baai/zl/todo/11.4/processing
cp pipeline115.example.json /home/baai/zl/todo/11.5/my_run.json
# 编辑my_run.json，将output_root设为本次专用空目录
bash pipeline115.sh check --config /home/baai/zl/todo/11.5/my_run.json
```

仅在环境未继承密钥时，通过不回显输入设置。不要把密钥值写进配置、Git、命令行参数或文档：

```bash
read -rsp 'Astra API key: ' ZYAPI_API_KEY
export ZYAPI_API_KEY
read -rsp 'LAN Qwen API key: ' QWEN_API_KEY
export QWEN_API_KEY

# 只查询远端模型身份；不启动本机服务、不做业务推理
bash pipeline115.sh check --config /home/baai/zl/todo/11.5/my_run.json --probe

# 首先建议单条代表数据；前台运行，日志独立落盘
bash pipeline115.sh run --config /home/baai/zl/todo/11.5/my_run.json \
  --only cs_0001_session_20261030_183844

# 查看状态与验证，均不调用模型
bash pipeline115.sh status --config /home/baai/zl/todo/11.5/my_run.json
bash pipeline115.sh validate --config /home/baai/zl/todo/11.5/my_run.json
```

去掉`--only`处理清单全部条目；可重复传入`--only`选择多个精确ID。同一运行目录的选择列表必须保持一致。当前样例仍指向既有29条输入清单，并不自动发现任意目录的新数据。新批次需要同字段结构的原始/三帧登记清单、逐条预处理配置和独立输出根目录；不能只换目录名就假定映射正确。

只读复用已完成预处理用于联调：

```bash
bash pipeline115.sh run --config /home/baai/zl/todo/11.5/my_run.json \
  --only cs_0001_session_20261030_183844 \
  --reuse-preprocessed-state /home/baai/zl/todo/11.4/processing/state.json
```

此模式重新核对原始三帧与视频、旧预处理审计中绑定的原始采集哈希，并验证五个预处理阶段完整文件哈希，不修改历史预处理，不访问bag；已有深度不占GPU重算。必须确认同一个原始采集目录，不能跨场景复用同名结果。

## 3. 输出与直接输入契约

`output_root`下：

```text
run_manifest.json          固定代码/配置/清单/事实身份及本次精确episode范围
config.json / state.json   配置快照与分阶段状态
rows/                     逐条审计后的三帧与原始文件哈希清单
configs/ / prepared/      本次有效预处理配置与输出引用
preprocessed/             新CPU与深度结果，不含ros_output
preprocessed_recovery/    只按冻结策略生成的独立恢复结果
frozen_runners/           恢复代码副本与原阈值策略
logs/ / receipts/         每次尝试日志、子进程身份和阶段输出哈希
resources/                GPU所有者、本次模型身份、启动及停止回执
annotations/
  target.schema.json
  episodes/<id>/
    input/                直接消费包：ready.json、all_frames.jsonl、RGB等
    selection/            Astra候选、粗筛、精筛和场景模型证据
    events/ / temporal/   逐事件响应及顺序图更新
    complete.json
  json/                   基础视觉JSON；未知结果仍unknown
  exports/<id>/           基础两个目标格式
  confirmed_json/         仅明确列入人工事实作用域的数据
  confirmed_exports/<id>/
  confirmed_receipts/
final_validation.json      最终独立校验结果
```

直接包`ego.direct_frames.v1`包含相机参数、VIO原接受报告和每帧JSONL。每帧保留原视频帧号、标签/fps时间、传感器时间、左右目解码索引、稳定`:direct:<序号>`ID、JPEG图及哈希、尺寸、相机坐标系、米制TF、协方差派生质量、完整原毫米位姿/协方差、深度/置信度清单及其文件引用/哈希。右图、原校正左图、深度与置信度保留在预处理目录，不能删除；这是有完整性引用的磁盘包，不是可脱离上游目录单独搬走的自包含档案。

JPEG沿用原bag边界的OpenCV质量90编码，没有降低分辨率或再次压缩旧JPEG。几何从mm转换m，协方差按6×6单位变换，与原ROS边界一致。深度为uint16/mm、置信度uint8。发布前检查图像类型尺寸、索引集、同步、严格递增帧号/时间、有限位姿及哈希；临时目录验证后原子发布ready.json。下游须验证ready及所引用上游文件。包完整通过不代表VIO质量通过，也不声称完成ROS反序列化验证。

事件原图继续独立读取左右三帧登记资料，保留标签和偏差审计；左目仍按相关≥.99、MAE≤4、±90帧做内容匹配；右目只检查哈希与解码，不能写成双目都做像素匹配。

## 4. 并发、背压与GPU所有权

初始CPU上限2、深度1、Astra1、每个独立远端接口1，本机请求1。示例队列上限4条，活跃直接包40GiB，可用内存门槛8GiB、剩余磁盘20GiB；这些是保守可配置初值，尚无整批峰值/吞吐调优结论。已完成文件保留，不以自动删除证据解决磁盘压力。

2026-11-05 后续新增可选整条调度：`dispatch: {"mode":"episode","endpoints":["local","lan"]}`。Astra 后由每台 Qwen 分别领取一条数据，并连续承担该条场景、事件、时序。端点必须能承接所有阶段的完整请求，实际每条路由及归属独立落盘。当前 72 条拟使用 8 条活跃队列、2 条深度分组和每端一路。旧阶段路由配置仍保留兼容；详见 [整条调度说明](EPISODE_DISPATCH_11_5.md)。

GPU用本机共同排他锁覆盖深度任务或本机Qwen整个驻留周期，不仅锁HTTP请求。实际显存门槛仍为深度10000MiB、本机模型启动20000MiB；发现其他计算进程或非本次llama-server则报告资源阻塞，不终止它。CPU和远端可继续已领取的工作。不同输出目录也共用同一GPU锁。旧版/第三方程序不遵守新锁，因此每次取得锁仍检查真实进程和显存。

深度按组交替本机模型，示例每组最多4条、本机驻留预算900秒。当深度等待且预算已到，写入yield标记；模型客户端只在新的未缓存请求前让出，当前请求不被打断，成功缓存和顺序时序图保留。HTTP请求空闲后检查`/slots`、PID/create_time/完整命令，SIGTERM停止本次服务，确认退出后才能进入深度。不强杀超时服务，不接管旧服务。

## 5. 恢复与失败边界

同一输出根目录由OS排他锁保护，每条模型阶段另有锁。运行绑定代码、配置、清单、人工事实和端点身份，任何变化必须使用新输出根或显式迁移，不能把变化后缓存冒充原版本。阶段输出哈希被篡改时拒绝恢复。已完成阶段和同输入请求不重跑，时序缓存另绑定端点并保留前置图签名。

```bash
# 正常中断后同命令继续；只消费合法回执
bash pipeline115.sh run --config /home/baai/zl/todo/11.5/my_run.json \
  --only cs_0001_session_20261030_183844

# 检查日志并修复失败原因后，才显式恢复失败阶段
bash pipeline115.sh run --config /home/baai/zl/todo/11.5/my_run.json \
  --only cs_0001_session_20261030_183844 --retry-failed
```

SIGINT/SIGTERM会停止领取新任务，等待已开始的工作到安全边界，记录stopped；正常停止可用同命令恢复。资源阻塞原因保留，须显式传入`--retry-failed`才重试。

使用只读复用模式时恢复仍须带原`--reuse-preprocessed-state`。本次联调曾发现自有服务正常退出后的TCP TIME_WAIT导致端口预检查误报，已修复为与HTTP服务相同的SO_REUSEADDR检查，活动监听仍会拒绝。测试目录内仅针对这一文件做了显式迁移：保存原清单/原代码/前后哈希，确认所有进程已退出且完成回执哈希全部通过；审计在`validation/live_astra/migrations/001_socket_time_wait/`。一般运行不会自动忽略代码身份变更。

进程消失不算成功；完成以真实阶段产物及哈希回执为准。检测仍活着的旧子进程时拒绝重复启动。资源阻塞后先释放外部任务或排查本次遗留服务，再恢复。

传输超时不能证明服务已结束。Qwen客户端保留尝试与“server completion unknown”，不做自动传输重试；恢复前确认远端原请求状态，不能仅凭超时立即重复提交。服务提供幂等键不等于已证明支持去重。容量失败不进入时序的语义修复重试；真正语义一致性错误沿用最多3次原策略。Astra沿用冻结的有界传输/模型身份验证。

协方差错误只在日志明确出现`pose covariance is materially indefinite`且未做过恢复时进入冻结策略；核对原/修改helper哈希，保留原阈值、排除坏样本、禁止跨坏样本插值，另写恢复配置/输出，保存原日志。其他错误不自动“修复”为成功。

## 6. 最终JSON与人工事实

继续使用[既有输出规范](OUTPUT_SPEC.md)和严格契约。保留旧81条内容范围、空间树的节点/父关系/未定位节点、逐事件空间变化、时序有效区间、证据三帧、实验字段和潜在搜索区域。未实测的参与者/搬移记录/测量继续空缺；不让模型伪造。状态枚举中历史`local_qwen...`词汇为契约兼容保留，实际来源看metadata及stage_profile/request receipts，不能据枚举断言来自本机。

基础json保留unknown。仅facts文件明确列出的当前29条按用户确认投影为前面not_found、最后found，保留视觉原判断；不外推到81条或未来任意输入，不推导无重访、精确找到帧或位置。两份end_version目标格式均由相同既有确定性导出器生成。

## 7. 已完成与未完成验收

已完成证据保存于`/home/baai/zl/todo/11.5/validation/`：

- 474帧真实预处理结果直接转换，与历史bag提取结果逐帧对比，JPEG字节、帧号、时间、TF、位姿全部一致；VIO原未通过状态保留。`direct_equivalence/equivalence_report.json`。
- 真实本机Qwen启动、结构化请求、空闲检查、正常卸载已通过；未停止其他服务。`local_residency/report.json`。
- 新目录中实际重新执行CPU预处理及GPU深度，CPU约47.50秒、深度约136.61秒，无ros_output或bag输出；直接包完整校验通过。此耗时不含Astra或Qwen业务标注。`fresh_preprocessing/report.json`。
  该次重新运行VIO后得到434帧，历史为474帧，VIO质量报告仍未通过；保持算法与阈值不保证VIO重新执行逐帧确定性。474帧字节等价结论仅针对同一份固定预处理结果的两种传输转换，不能扩大为新旧完整推理逐字相同。
- 一个完整4事件样本严格比对9次历史请求的prompt/schema/图像字节/预算后回放，走新场景、事件、顺序时序与最终组装，事件和空间时序内容与旧版一致，两个目标导出及限定人工事实通过。此项没有新推理。`replay_report.json`。
- 21项自动测试覆盖错误模型、截断、schema、缓存篡改、配置变化、重定向、密钥脱敏、容量门槛、请求边界让出、OS排他锁、外部GPU拒绝、直接包坏时间/类型拒绝、单位变换、中断排空、资源阻塞恢复门槛、旧预处理原始采集变更拒绝、真实套接字TIME_WAIT/活动监听区分和流水线阶段顺序/互斥；旧11-4实际组装回归亦通过。
- 新地址鉴权模型清单确认Q8_0及8192上下文，`lan_identity.json`。真实远端4次事件调用已完成，阶段约165.76秒；`live_astra/live_events_report.json`。
- 434帧新直接包的Astra粗筛/精筛真实调用完成，阶段873.93秒；粗筛经历超时及HTTP 500后第三次成功，精筛一次成功。此耗时包含服务错误重试，不代表正常延迟；`live_astra/live_astra_report.json`。

单条分阶段真实联调已完成：本机场景新推理约158.04秒；4次顺序时序更新128.85秒；调度器复用此前在本次测试中完成的真实CPU/深度、Astra、事件产物，最终1/1通过直接包、schema、空间树/时序、原始证据、哈希和两种导出验证。人工事实投影亦已生成，本次自有模型正常卸载。`live_astra/final_validation.json`及`scheduler_resume_report.json`。

这是同一个样本的分阶段真实联调，包含一次修复后断点恢复，不能用调度器最后一轮133.69秒作为整条数据从零开始的处理时间。阶段求和也不等于批处理墙钟时间；测试中远端事件与Astra存在重叠。全部推理结果仍属未人工复核的模型标注。

尚未完成：连续冷启动整条基准、多条实际并发吞吐和峰值资源测量、更多故障及公平性交接压力测试。本版本以`11-5-dev`标记开发验收状态，单条通过不等于29条批处理或生产验收完成，不承诺整体提速。提交内的证据索引见[VALIDATION_11_5.json](VALIDATION_11_5.json)。

测试命令：

```bash
python3 -m unittest discover -s tests115 -v
PYTHONPATH=. python3 tests115/replay_episode.py
python3 test_release.py
bash pipeline115.sh check
```

回放测试依赖当前机器保存的历史证据，明确不等价于离线模型模拟或新标注。实测运行时间见各报告，不能把一次短文本启动测试用作整条数据处理时间。
