# 用自己的电脑和 Cloudflare Tunnel 开放内测

[English](cloudflare-tunnel.md) | **简体中文**

这套部署让别人通过 `https://paperleet.com` 访问你电脑上的 PaperLeet，暂时不需要租服务器或设置路由器端口转发。Cloudflare 提供公网 HTTPS 入口；网页、账号、数据库和隔离判题服务在你的电脑上通过 Docker 运行。内测期间，需要电脑保持开机、联网、不休眠，并运行 Docker。

```text
浏览器 → paperleet.com → Cloudflare → cloudflared 容器 → 网页 → API
                                                           ├→ 持久数据卷
                                                           └→ 判题管理服务 → 隔离 worker
```

`make dev` 用于个人开发。这套配置使用 hosted 模式、独立测试账号和容器判题，所有服务都不向宿主机发布端口。Tunnel 连接网页容器，不连接开发端口 5173 或 8765。

## 1. 准备 Docker 和域名

启动 Mac 上的 Docker Desktop，确认以下命令正常：

```sh
docker version
docker compose version
```

需要 Docker Compose 2.24.4 或更新版本。给 Docker 分配至少 4 GB 内存；默认同时运行一个判题任务，每个 worker 上限 2 GB。首次构建会下载 Python、PyTorch 和前端依赖，可能需要几分钟。

在 Cloudflare 控制台确认 `paperleet.com` 已在你的账号中处于 Active 状态。通过 Cloudflare 购买的域名已经使用其域名服务器；其他注册商购买的域名按[官方接入步骤](https://developers.cloudflare.com/dns/zone-setups/full-setup/setup/)设置。

## 2. 创建 Tunnel

在 Cloudflare 账号中打开 **Networking → Tunnels → Create Tunnel**。部分控制台位于 **Zero Trust → Networks → Tunnels**。如需选择连接器，选择 `cloudflared`，名称填 `paperleet-beta`。

在安装说明中选择 **Docker**，复制示例命令中 `--token` 后面的 token。仓库会负责启动连接器，不需要再单独执行 Cloudflare 展示的 Docker 命令。token 请自己保管。

参考[官方 Tunnel 创建说明](https://developers.cloudflare.com/tunnel/get-started/)。

## 3. 配置并启动 PaperLeet

在仓库根目录运行：

```sh
cp -n deploy/tunnel.env.example deploy/.env
chmod 600 deploy/.env
openssl rand -hex 32
```

`cp -n` 会保留已经存在的配置。用编辑器打开 `deploy/.env`，填写下面这些值；判题 token 使用最后一条命令生成的随机字符串：

```dotenv
PAPERLEET_DOMAIN=paperleet.com
PAPERLEET_AUTH_MODE=test
TUNNEL_TOKEN=这里填Cloudflare的TunnelToken
PAPERLEET_JUDGE_TOKEN=这里填生成的64位十六进制字符串
SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
PAPERLEET_INVITE_EMAILS=
PAPERLEET_ADMIN_EMAILS=
```

Cloudflare token 和判题 token 是两份不同的凭据。`deploy/.env` 已被 Git 和镜像构建排除，不要提交或把内容贴到聊天、issue 中。

```sh
./deploy/tunnel.sh build
./deploy/tunnel.sh test-users create tester1 --name "测试用户 1"
./deploy/tunnel.sh up
./deploy/tunnel.sh status
```

保存创建账号时输出的用户名和访问码，每位参与者创建一个独立账号。以前通过本地 Python 命令创建的账号位于 `.paperleet/`，不会自动复制到这套 Docker 部署，需要在这里单独创建。

脚本默认使用独立的 `paperleet-tunnel` Compose 项目，与直接部署到服务器的版本分开。管理这套内测都使用 `tunnel.sh`；如果显式设置 `COMPOSE_PROJECT_NAME`，后续操作始终使用同一个值。

## 4. 绑定 paperleet.com

等 Cloudflare 控制台显示连接器 **Healthy** 后，在 Tunnel 的 **Routes** 中添加 **Published application**：

| 设置 | 填写内容 |
| --- | --- |
| Subdomain / 子域名 | 留空 |
| Domain / 域名 | `paperleet.com` |
| Path / 路径 | 留空 |
| Service type / 服务类型 | `HTTP` |
| Service URL / 服务地址 | `web:80`，完整地址为 `http://web:80` |

保持原始公网 Host，HTTP Host Header 的覆盖选项留空。`web` 是 Docker 服务名称；这里不能填 `localhost`，否则会指向连接器容器自己。

保存路由后，正在运行的应用就会通过公网域名开放。Cloudflare 会创建对应 DNS 路由，不需要把家里的 IP 填进 A 记录。如果已有同名记录，先处理该条 DNS 冲突。这里使用根域名，不会自动配置 `www`。

浏览器访问使用 HTTPS。上面的 HTTP 连接位于加密隧道终止后的 Docker 内部，因此 Caddy 不需要为这套配置申请公网证书，也不需要开放入站端口。

## 5. 验证访问和答题

从另一台电脑，或关闭 Wi-Fi 使用移动网络的手机打开 `https://paperleet.com`，用测试账号登录，保存草稿并运行一道题。也可以检查：

```sh
curl --fail https://paperleet.com/api/health
./deploy/tunnel.sh logs
```

健康响应应包含 `"mode":"hosted"`、`"auth_required":true` 和 `"judge":{"ready":true,...}`。仅 Docker 显示运行中，不代表公网域名和 Tunnel 已经连通。

Tunnel 配置将单次判题 worker 上限设为 90 秒，API 最多等待 110 秒，为 [Cloudflare 的代理超时](https://developers.cloudflare.com/fundamentals/reference/connection-limits/)留出余量。题目测试内容不变，运行过慢或死循环会触发这个限制。

## 停止、继续、更新和备份

```sh
# 关闭公网访问和应用，保留数据
./deploy/tunnel.sh down

# 重新开放
./deploy/tunnel.sh up

# 备份账号、帖子、草稿和判题记录
./deploy/tunnel.sh backup

# 备份后更新
git pull --ff-only
./deploy/tunnel.sh build
./deploy/tunnel.sh up
```

用户数据保存在 Docker 的 `app_data` 数据卷中，不会保存到 GitHub 或 Cloudflare。私有备份文件位于 `deploy/backups/`，请另外复制一份到其他私有位置。重新构建或执行 `down` 会保留数据卷；删除数据卷或重置 Docker Desktop 前需要先备份。恢复流程见[部署文档](hosting.zh-CN.md#备份与恢复)，管理这套部署时将命令替换为 `tunnel.sh`。

临时内测期间，可在单独的 Mac 终端运行 `caffeinate -i` 防止空闲休眠，按 Ctrl-C 结束。保持屏幕盖打开；合盖休眠、断网、退出 Docker 或关机都会中断访问。

如果 Tunnel 不健康，检查 Docker 和 `tunnel.sh logs`。遇到 502，先确认服务地址是 `http://web:80`，以及网页容器是否启动；遇到 403，检查访问域名与 `PAPERLEET_DOMAIN` 是否一致。保持 `/api/*` 不缓存，并关闭 Cloudflare Pseudo IPv4 的覆盖请求头模式，保留正常账号请求行为。
