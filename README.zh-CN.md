# PaperLeet 离线版

[English](README.md) | **简体中文**

读论文、实现核心方法，在自己的 CPU 上运行测试。这个版本只保留题库、Python 编辑器、本地解答、测试报告和教学实验，不包含账号、个人主页、社区、在线人数或云端提交。

题库跟随 [PaperLeet 公共主干](https://github.com/PaperLeet/paperleet.github.io)，本次整理包含 **49 篇论文、63 道练习**。`sync/` 记录同步的上游版本及文件清单。

## 本地启动

需要 Python 3.10+ 和 Node.js 20.19+。首次安装依赖需要联网；安装后正常做题只使用本机资源。

```bash
git clone https://github.com/Hongyang-Du/PaperLeet.git
cd PaperLeet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
npm --prefix web ci
make dev
```

打开 **http://127.0.0.1:5173**，选择论文和模块，补全代码并运行测试。Python/PyTorch 在自己的电脑上执行，不需要 GPU，也不需要登录。只有主动打开论文链接时才会访问外部网站。

下次直接运行 `make dev`，**Ctrl-C 会同时停止网页和判题服务**。服务只绑定 `127.0.0.1`，一次执行一个判题任务。Python 代码使用当前本机用户的权限运行，请用于自己的练习代码。

## 用自己的编辑器做题

```bash
paperleet list
paperleet init architecture/rms_norm
paperleet test architecture/rms_norm
```

`workspace/` 保存解答，`reports/` 保存测试结果，都不会提交到 Git。更新题库时保留这些目录。网页也保存浏览器恢复草稿，不会把代码发送给 PaperLeet 在线服务。

## 自动更新题库

[同步 workflow](.github/workflows/sync-upstream.yml) 每天检查公共主干，也可在 **Actions** 中手动运行。有变化时，离线检查通过后创建或更新同一个同步 PR，供维护者合并。

只同步题目包、论文信息、必要的判题公共代码和已生成的实验数据。离线界面和本地服务由这里维护，不会同步账号、社区、线上部署、密钥或用户解答。

详见[同步说明](docs/upstream-sync.md)。同步 PR 合并后，在自己的电脑执行 `git pull`，重新启动 `make dev` 即可使用新题库。

## 参与和检查

题目改进通常提交到[公共主干](https://github.com/PaperLeet/paperleet.github.io)，让在线、离线两版共享修复；离线界面和本机运行方式的改进提交到这里。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

```bash
make check   # 框架和前端检查
make build   # 导出题库并构建网页
make ci      # 额外验证参考解、错误实现和 CPU 时间预算
```

Git 历史保留了以前的合并应用；当前版本是独立离线版。

## 许可证

采用 [MIT 许可证](LICENSE)。题包原有的来源署名和第三方声明继续保留。
