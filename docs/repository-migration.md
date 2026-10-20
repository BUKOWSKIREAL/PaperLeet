# 本地产品与公共核心迁移

本仓已有独立本地前端、CLI 和本机 Python/PyTorch 判题服务，继续作为开源
本地产品，不依赖官方账号、生产凭据或私有平台。

由 owner 将现有仓库直接 Transfer 为 `PaperLeet/paperleet`，保留 Star、
Issues、PR 和历史。迁移前不要创建同名替代仓，迁移后不要重建旧地址。

公共题库、数学测试、判题核心与浏览器运行能力目前仍在
`PaperLeet/paperleet.github.io` 开发。整合完成前保持现有白名单同步的上游和
工作流。受管理文件先向上游贡献；本地前端和服务改进在本仓评审。

后续保留本地界面和文件管理能力，整合公共核心及 WASM 接入，并保留原始
开发提交可追溯性。原有 github.io PR、讨论和评审继续在那里查阅。

私有平台通过固定公共 commit 采用题库和判题改进，不会覆盖其私有线上前端。
TorchWasm（原 StrideTorch）独立维护，产品采用经过验证的版本。GitHub 独立仓库
目前仍是 `PaperLeet/stridetorch`；公共主分支现已接入改名为 `torchwasm` 的
0.91；独立仓库的[接入 PR](https://github.com/PaperLeet/stridetorch/pull/1) 保留原生与 WASM 验证记录。运行时改名需要公共包路径、导入、浏览器 Worker、
资源和来源锁文件一起更新；本仓的白名单同步继续跟随已审核的公共快照，不能越过现有同步边界。
仓库 URL 改名和 owner 转移分别处理，未发生前保留可用上游。

迁移验收包括干净安装、本地完整练习、浏览器与原生执行、线上平台集成。
GitHub 迁移不涉及用户 workspace、报告、浏览器草稿或生产数据库。
