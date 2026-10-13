import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  LogOut,
  Mail,
  ShieldCheck,
} from "lucide-react";
import { useI18n } from "./i18n";
import {
  Avatar,
  InstitutionBadge,
  SocialNotice,
  errorMessage,
  profileHref,
  socialApi,
  useAccount,
} from "./social";
import type { AccountUser } from "./social";

const COLORS = [
  "#4f7c73",
  "#8665a1",
  "#5b78a6",
  "#ac7545",
  "#a45e78",
  "#627a43",
  "#747789",
  "#9b6853",
];
export default function Account() {
  const { user, loading, capabilities, error, refresh } = useAccount();
  const { t } = useI18n();
  return (
    <main className="social-page account-page">
      <a
        className="social-back"
        href={user ? profileHref(user) : "#/community"}
      >
        <ArrowLeft size={14} />
        {user
          ? t("返回个人主页", "Back to profile")
          : t("返回社区", "Back to community")}
      </a>
      {loading ? (
        <div className="social-loading">
          <span className="spinner" />
          {t("正在加载账户…", "Loading account…")}
        </div>
      ) : error ? (
        <>
          <SocialNotice error={error} />
          <button className="button" onClick={() => void refresh()}>
            {t("重试", "Retry")}
          </button>
        </>
      ) : user ? (
        <ProfileSettings key={user.id} user={user} />
      ) : (
        <SignIn enabled={capabilities?.auth_available === true} />
      )}
    </main>
  );
}
function SignIn({ enabled }: { enabled: boolean }) {
  const { t, locale } = useI18n();
  const { setUser, capabilities } = useAccount();
  const [email, setEmail] = useState(""),
    [code, setCode] = useState(""),
    [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  const domain = email.trim().toLowerCase().split("@")[1];
  const institution = capabilities?.institutions.find((i) =>
    i.domains.includes(domain),
  );
  async function sendCode() {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await socialApi("account/send-code", { email: email.trim() });
      setSent(true);
      setMessage(
        t(
          "验证码已发送，请检查机构邮箱。",
          "A verification code has been sent to your institutional email.",
        ),
      );
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function verify() {
    setBusy(true);
    setError("");
    try {
      const data = await socialApi<{ user: AccountUser }>("account/verify", {
        email: email.trim(),
        code: code.trim(),
      });
      setUser(data.user);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="account-grid">
      <section className="social-panel login-card">
        <span className="social-eyebrow">PAPERLEET ACCOUNT</span>
        <h1>{t("加入研究者社区", "Join the community")}</h1>
        <p className="social-description">
          {t(
            "用学校或机构邮箱登录，与大家讨论论文和实现。",
            "Sign in with your school or institutional email to discuss papers and implementations.",
          )}
        </p>
        {!enabled && (
          <SocialNotice>
            {t(
              "此部署尚未启用邮箱登录。你可以浏览社区；发帖和关注需在管理员配置邮件验证后使用。",
              "Email sign-in is not enabled on this deployment yet. You can browse the community; posting and following will be available after the administrator configures verification.",
            )}
          </SocialNotice>
        )}
        {error && <SocialNotice error={error} />}
        {message && <SocialNotice>{message}</SocialNotice>}
        <form
          className="social-form"
          onSubmit={(e) => {
            e.preventDefault();
            void (sent ? verify() : sendCode());
          }}
        >
          <label>
            {t("学校／机构邮箱", "School / institutional email")}
            <input
              type="email"
              autoComplete="email"
              required
              maxLength={254}
              value={email}
              disabled={busy || sent || !enabled}
              placeholder="you@university.edu"
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          {institution && (
            <div className="recognized-institution">
              <ShieldCheck size={14} />
              {locale === "en"
                ? institution.name_en || institution.name
                : institution.name}
            </div>
          )}
          {sent && (
            <label>
              {t("邮件验证码", "Email verification code")}
              <input
                autoComplete="one-time-code"
                inputMode="numeric"
                required
                value={code}
                maxLength={12}
                autoFocus
                disabled={busy}
                onChange={(e) => setCode(e.target.value)}
                placeholder={t(
                  "输入邮件中的验证码",
                  "Enter the code from your email",
                )}
              />
            </label>
          )}
          <button className="button" type="submit" disabled={busy || !enabled}>
            {busy ? (
              <span className="spinner" />
            ) : sent ? (
              <ShieldCheck size={15} />
            ) : (
              <Mail size={15} />
            )}
            {sent
              ? t("验证并登录", "Verify and sign in")
              : t("发送验证码", "Send verification code")}
          </button>
          {sent && (
            <div className="form-inline-actions">
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  setSent(false);
                  setCode("");
                  setMessage("");
                }}
              >
                {t("更换邮箱", "Change email")}
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => void sendCode()}
              >
                {t("重新发送", "Resend code")}
              </button>
            </div>
          )}
        </form>
        <p className="account-fineprint">
          {t(
            "只有通过审核的机构邮箱可以注册。昵称和头像始终公开，机构和国家／地区由你选择是否展示。",
            "Registration is limited to approved institutional email domains. Your display name and avatar are always public. You choose whether to display your institution and country or region.",
          )}
        </p>
      </section>
      <aside className="account-aside">
        <div className="social-panel">
          <ShieldCheck size={27} className="accent-icon" />
          <h3>{t("机构邮箱验证", "Institutional email verification")}</h3>
          <p>
            {t(
              "验证说明你拥有这个机构邮箱，不代表在读或在职身份认证。完整邮箱不会出现在公开主页。",
              "Verification confirms access to an institutional mailbox. It does not certify current student or employment status. Your full email address is never shown on your public profile.",
            )}
          </p>
        </div>
        <InstitutionRequest />
      </aside>
    </div>
  );
}
function InstitutionRequest() {
  const { t } = useI18n();
  const [name, setName] = useState(""),
    [website, setWebsite] = useState(""),
    [domain, setDomain] = useState("");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [done, setDone] = useState(false);
  async function submit() {
    setBusy(true);
    setError("");
    try {
      await socialApi("account/institution-requests", {
        name: name.trim(),
        website: website.trim(),
        email_domain: domain.trim(),
      });
      setDone(true);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="social-panel institution-request">
      <summary>
        {t("找不到你的机构？", "Institution not listed?")}
        <ChevronDown size={15} />
      </summary>
      <p>
        {t(
          "提供机构信息，管理员审核域名后即可使用邮箱验证。",
          "Suggest your institution. Email verification becomes available once an administrator approves its domain.",
        )}
      </p>
      {error && <SocialNotice error={error} />}
      {done ? (
        <SocialNotice>
          {t(
            "申请已提交，正在等待管理员审核。",
            "Your request has been submitted for administrator review.",
          )}
        </SocialNotice>
      ) : (
        <form
          className="social-form"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <label>
            {t("机构名称", "Institution name")}
            <input
              required
              maxLength={160}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label>
            {t("官方网站", "Official website")}
            <input
              type="url"
              required
              maxLength={500}
              placeholder="https://university.edu"
              value={website}
              onChange={(e) => setWebsite(e.target.value)}
            />
          </label>
          <label>
            {t("邮箱域名", "Email domain")}
            <input
              required
              maxLength={253}
              placeholder="university.edu"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
            />
          </label>
          <button className="button subtle" disabled={busy}>
            {t("申请收录", "Request review")}
            <ArrowRight size={14} />
          </button>
        </form>
      )}
    </details>
  );
}
function ProfileSettings({ user }: { user: AccountUser }) {
  const { t } = useI18n();
  const { setUser } = useAccount();
  const [name, setName] = useState(user.display_name),
    [username, setUsername] = useState(user.username),
    [color, setColor] = useState(user.avatar_color),
    [bio, setBio] = useState(user.bio || "");
  const [country, setCountry] = useState(user.country || ""),
    [showInstitution, setShowInstitution] = useState(user.show_institution),
    [showCountry, setShowCountry] = useState(user.show_country),
    [showProgress, setShowProgress] = useState(user.show_progress);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  useEffect(() => {
    setMessage("");
  }, [
    name,
    username,
    color,
    bio,
    country,
    showInstitution,
    showCountry,
    showProgress,
  ]);
  async function save() {
    setBusy(true);
    setError("");
    try {
      const data = await socialApi<{ user: AccountUser }>("account/profile", {
        display_name: name.trim(),
        username: username.trim(),
        avatar_color: color,
        bio,
        country: country.trim().toUpperCase(),
        show_institution: showInstitution,
        show_country: showCountry,
        show_progress: showProgress,
      });
      setUser(data.user);
      setMessage(t("个人资料已保存。", "Your profile has been saved."));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function logout() {
    setBusy(true);
    setError("");
    try {
      await socialApi("account/logout", {});
      setUser(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="social-page-heading">
        <div>
          <span className="social-eyebrow">YOUR ACCOUNT</span>
          <h1>{t("个人资料与隐私", "Profile & privacy")}</h1>
        </div>
        <a className="button subtle" href={profileHref(user)}>
          {t("查看公开主页", "View public profile")}
          <ArrowRight size={14} />
        </a>
      </div>
      <div className="account-grid">
        <form
          className="social-panel social-form profile-settings"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          {error && <SocialNotice error={error} />}
          {message && <SocialNotice>{message}</SocialNotice>}
          <div className="settings-avatar">
            <Avatar
              user={{ display_name: name, avatar_color: color }}
              size={64}
            />
            <div>
              <h3>{t("公开头像", "Public avatar")}</h3>
              <div className="avatar-palette">
                {COLORS.map((c) => (
                  <button
                    key={c}
                    type="button"
                    aria-label={`${t("头像颜色", "Avatar color")} ${c}`}
                    aria-pressed={color === c}
                    style={{ backgroundColor: c }}
                    onClick={() => setColor(c)}
                  >
                    {color === c && <Check size={14} />}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <label>
            {t("公开昵称", "Public display name")}
            <input
              required
              minLength={1}
              maxLength={40}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label>
            {t("用户名", "Username")}
            <input
              required
              minLength={3}
              maxLength={24}
              pattern="[a-z0-9_]{3,24}"
              value={username}
              onChange={(e) => setUsername(e.target.value.toLowerCase())}
            />
            <small>
              {t(
                "3–24 位小写字母、数字或下划线，用于个人主页链接。",
                "3–24 lowercase letters, numbers, or underscores. Used in your profile link.",
              )}
            </small>
          </label>
          <label>
            {t("个人简介", "Bio")}
            <textarea
              rows={3}
              maxLength={500}
              value={bio}
              onChange={(e) => setBio(e.target.value)}
            />
          </label>
          <div className="settings-section">
            <h3>{t("公开资料", "Public visibility")}</h3>
            <p>
              {t(
                "昵称和头像始终公开。帖子、回复、点赞、转发和关注关系会展示在你的主页。",
                "Your display name and avatar are always public. Posts, replies, likes, reposts, and follow connections appear on your profile.",
              )}
            </p>
          </div>
          <label className="social-checkbox">
            <input
              type="checkbox"
              checked={showInstitution}
              onChange={(e) => setShowInstitution(e.target.checked)}
            />
            <span>
              {t("显示已验证机构", "Show my verified institution")}
              <small>{user.institution?.name}</small>
            </span>
          </label>
          <label>
            {t("国家／地区代码（选填）", "Country / region code (optional)")}
            <input
              maxLength={2}
              pattern="[A-Za-z]{2}|"
              value={country}
              placeholder="US / CN / GB"
              onChange={(e) => setCountry(e.target.value.toUpperCase())}
            />
            <small>
              {t(
                "由你填写两位国家代码，不根据 IP 自动发布位置。",
                "Enter a two-letter country code. Your location is not published automatically from your IP.",
              )}
            </small>
          </label>
          <label className="social-checkbox">
            <input
              type="checkbox"
              checked={showCountry}
              onChange={(e) => setShowCountry(e.target.checked)}
            />
            <span>{t("显示我的国家／地区", "Show my country or region")}</span>
          </label>
          <label className="social-checkbox">
            <input
              type="checkbox"
              checked={showProgress}
              onChange={(e) => setShowProgress(e.target.checked)}
            />
            <span>
              {t("允许展示学习进度", "Allow learning progress to be public")}
              <small>
                {t(
                  "当前练习进度仍保存在本机，尚未同步到公开主页。",
                  "Practice progress is currently saved in this browser and is not yet synced to your public profile.",
                )}
              </small>
            </span>
          </label>
          <div className="form-footer">
            <button className="button" type="submit" disabled={busy}>
              {busy ? <span className="spinner" /> : <Check size={15} />}
              {t("保存设置", "Save changes")}
            </button>
          </div>
        </form>
        <aside className="account-aside">
          <section className="social-panel">
            <h3>{t("已验证的账户", "Verified account")}</h3>
            <p className="private-email">{user.email}</p>
            <InstitutionBadge institution={user.institution} />
            <p>
              {t(
                "此处的完整邮箱仅本人可见。机构名称是否公开由左侧开关控制。",
                "Only you can see your full email here. Use the visibility setting to control whether your institution appears publicly.",
              )}
            </p>
            <button
              className="button subtle logout-button"
              type="button"
              disabled={busy}
              onClick={() => void logout()}
            >
              <LogOut size={14} />
              {t("退出登录", "Sign out")}
            </button>
          </section>
          <section className="social-panel">
            <h3>{t("公开预览", "Public preview")}</h3>
            <div className="profile-preview">
              <Avatar
                user={{ display_name: name, avatar_color: color }}
                size={46}
              />
              <b>{name || t("你的昵称", "Your display name")}</b>
              <span>@{username}</span>
              {showInstitution && (
                <InstitutionBadge institution={user.institution} />
              )}
              {showCountry && country && <span>{country}</span>}
              <p>{bio}</p>
            </div>
          </section>
        </aside>
      </div>
    </>
  );
}
