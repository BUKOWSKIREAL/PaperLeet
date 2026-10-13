import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { ShieldCheck } from "lucide-react";
import { useI18n } from "./i18n";

export type Institution = { id: string; name: string; name_en?: string };
export type PublicUser = {
  id: string;
  username: string;
  display_name: string;
  avatar_color: string;
  bio: string;
  created_at: string;
  institution?: Institution;
  country?: string;
  show_progress?: boolean;
};
export type AccountUser = PublicUser & {
  email: string;
  role: string;
  show_institution: boolean;
  show_country: boolean;
  show_progress: boolean;
  institution: Institution;
  country: string;
};
export type AccountCapabilities = {
  auth_available: boolean;
  institution_verification_required: boolean;
  institutions: (Institution & { domains: string[] })[];
  reason?: string;
};
export class SocialError extends Error {
  status: number;
  code?: string;
  constructor(message: string, status: number, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}
export async function socialApi<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    credentials: "same-origin",
    cache: "no-store",
    ...(body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new SocialError(
      "The account and community service is unavailable. Please try again shortly.",
      response.status,
    );
  }
  if (!response.ok)
    throw new SocialError(
      data.error || "Request failed",
      response.status,
      data.code,
    );
  return data as T;
}
type AccountContextValue = {
  user: AccountUser | null;
  capabilities?: AccountCapabilities;
  loading: boolean;
  error: string;
  refresh: () => Promise<void>;
  setUser: (user: AccountUser | null) => void;
};
const AccountContext = createContext<AccountContextValue>({
  user: null,
  loading: true,
  error: "",
  refresh: async () => {},
  setUser: () => {},
});
export function AccountProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AccountUser | null>(null);
  const [capabilities, setCapabilities] = useState<AccountCapabilities>();
  const [loading, setLoading] = useState(true),
    [error, setError] = useState("");
  async function refresh() {
    setLoading(true);
    try {
      const data = await socialApi<{
        user: AccountUser | null;
        capabilities: AccountCapabilities;
      }>("account/me");
      setUser(data.user);
      setCapabilities(data.capabilities);
      setError("");
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void refresh();
  }, []);
  return (
    <AccountContext.Provider
      value={{ user, capabilities, loading, error, refresh, setUser }}
    >
      {children}
    </AccountContext.Provider>
  );
}
export const useAccount = () => useContext(AccountContext);
export const errorMessage = (e: unknown) =>
  e instanceof Error ? e.message : "Something went wrong. Please try again.";
export const profileHref = (user: Pick<PublicUser, "username">) =>
  `#/u/${encodeURIComponent(user.username)}`;
export function Avatar({
  user,
  size = 34,
}: {
  user: Pick<PublicUser, "display_name" | "avatar_color">;
  size?: number;
}) {
  const safeColor = /^#[0-9a-f]{6}$/i.test(user.avatar_color)
    ? user.avatar_color
    : "#4f7c73";
  return (
    <span
      className="social-avatar"
      style={{
        width: size,
        height: size,
        backgroundColor: safeColor,
        fontSize: size * 0.36,
      }}
      aria-label={user.display_name}
    >
      {Array.from(user.display_name.trim())
        .slice(0, 1)
        .join("")
        .toLocaleUpperCase() || "P"}
    </span>
  );
}
export function PersonLink({
  user,
  compact = false,
}: {
  user: PublicUser;
  compact?: boolean;
}) {
  return (
    <a className="person-link" href={profileHref(user)}>
      <Avatar user={user} size={compact ? 24 : 34} />
      <span>{user.display_name}</span>
    </a>
  );
}
export function InstitutionBadge({
  institution,
}: {
  institution?: Institution;
}) {
  const { t, locale } = useI18n();
  return institution ? (
    <span
      className="institution-badge"
      title={t("机构邮箱已验证", "Institutional email verified")}
    >
      <ShieldCheck size={13} />
      {locale === "en"
        ? institution.name_en || institution.name
        : institution.name}
    </span>
  ) : null;
}
export function DateLabel({ value }: { value: string }) {
  const { locale } = useI18n();
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return (
    <time
      dateTime={value}
      title={date.toLocaleString(locale === "zh" ? "zh-CN" : "en-US")}
    >
      {date.toLocaleDateString(locale === "zh" ? "zh-CN" : "en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })}
    </time>
  );
}
export function AccountNav() {
  const { user, loading } = useAccount();
  const { t } = useI18n();
  return user ? (
    <a
      className="account-nav"
      href={profileHref(user)}
      title={t("我的主页", "My profile")}
    >
      <Avatar user={user} size={29} />
    </a>
  ) : (
    <a className="account-nav sign-in-nav" href="#/account">
      {loading ? "…" : t("登录", "Sign in")}
    </a>
  );
}
export function SocialNotice({
  error,
  children,
}: {
  error?: string;
  children?: ReactNode;
}) {
  return error ? (
    <div className="social-notice error" role="alert">
      {error}
    </div>
  ) : (
    <div className="social-notice" role="status">
      {children}
    </div>
  );
}
