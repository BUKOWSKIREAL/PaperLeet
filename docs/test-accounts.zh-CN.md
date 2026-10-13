# 内测测试账号

[English](test-accounts.md) | **简体中文**

内测可以使用管理员创建的测试账号。每位参与者拿到一个登录名和随机访问码，即可登录，不需要注册机构邮箱，也不需要 Supabase、邮件发送服务或机构审核。没有公开注册入口，也没有大家共用的默认密码。

## 线上内测

服务器、域名、Docker 和 HTTPS 按照[部署说明](hosting.zh-CN.md)准备。在 `deploy/.env` 中设置：

```dotenv
PAPERLEET_AUTH_MODE=test
SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
PAPERLEET_INVITE_EMAILS=
PAPERLEET_ADMIN_EMAILS=
```

保留实际的 `PAPERLEET_DOMAIN` 和随机生成的 `PAPERLEET_JUDGE_TOKEN`。测试模式只改变登录方式，线上判题仍使用隔离容器和 HTTPS。

构建部署，为每位测试者创建单独账号，然后启动：

```sh
./deploy/manage.sh build
./deploy/manage.sh test-users create tester1 --name "测试用户 1"
./deploy/manage.sh test-users create tester2 --name "测试用户 2"
./deploy/manage.sh up
```

每条创建命令会输出包含 `username` 和 `access_code` 的 JSON。把这两个值和网站地址发给对应测试者即可。访问码仅在创建或重置时显示，数据库只保存加盐哈希；代码仓库、前端页面和账号列表都不包含访问码。

测试者打开网站的**账号**页面，输入登录名和访问码即可登录。线上每个账号都有独立的私人草稿、判题历史和社区身份。个人主页与作者信息明确显示“**内测账号**”，不会显示为已验证机构身份。

## 管理账号

```sh
# 查看账号列表，不显示访问码
./deploy/manage.sh test-users list

# 生成新访问码，同时撤销旧会话
./deploy/manage.sh test-users reset-code tester1

# 停用账号，同时撤销会话
./deploy/manage.sh test-users disable tester2

# 可选：创建维护者账号；普通账号默认是 member
./deploy/manage.sh test-users create maintainer --name "维护者" --admin
```

登录名是固定的，不会随着公开用户名或昵称变化。测试账号、访问码哈希及其数据保存在应用数据卷中，也会进入私有备份。

## 先在本机体验登录

已有开发环境时，在同一个终端执行：

```sh
export PAPERLEET_AUTH_MODE=test
.venv/bin/python -m paperleet.test_accounts create tester1 --name "测试用户 1"
make dev
```

本地开发模式仍将练习草稿保存在浏览器，并在本机运行代码。需要验证按账号隔离的云端练习和多人隔离判题时，使用线上部署配置。若开发服务已经运行，修改认证模式后需要停止并重新启动。

## 切回机构邮箱登录

默认模式仍是 `institutional`。切回时设置 `PAPERLEET_AUTH_MODE=institutional`，补全 Supabase/SMTP、机构域名和邮箱邀请名单，再运行 `./deploy/manage.sh up`。机构邮箱模式下，测试账号及其会话无法使用；测试模式下，机构邮箱账号的会话也无法使用。

切换模式不会删除数据，也不会把测试账号转换成已验证机构账号。已有公开帖子仍保留“内测账号”标记。正式开放时若需要干净的数据，可使用新的部署数据卷，并按需私下备份内测数据。
