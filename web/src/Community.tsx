import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  BriefcaseBusiness,
  Check,
  ChevronLeft,
  ChevronRight,
  Code2,
  Coffee,
  Globe2,
  Heart,
  HelpCircle,
  MessageCircle,
  MessagesSquare,
  Plus,
  Repeat2,
  Search,
  Settings2,
  Trash2,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { useI18n } from "./i18n";
import {
  Avatar,
  DateLabel,
  InstitutionBadge,
  PersonLink,
  SocialNotice,
  errorMessage,
  profileHref,
  socialApi,
  useAccount,
} from "./social";
import type { PublicUser } from "./social";

type Counts = { likes: number; reposts: number; replies: number };
type Post = {
  id: string;
  title: string;
  body: string;
  category: string;
  created_at: string;
  updated_at: string;
  author: PublicUser;
  counts: Counts;
  viewer: { liked: boolean; reposted: boolean; can_delete: boolean };
};
type Reply = {
  id: string;
  post_id: string;
  parent_id: string | null;
  body: string;
  created_at: string;
  author: PublicUser;
  can_delete: boolean;
};
type PostDetail = Post & { replies: Reply[] };
type Profile = PublicUser & {
  counts: Counts & { posts: number; followers: number; following: number };
  viewer: { following: boolean; is_self: boolean };
};
type Connection = PublicUser & {
  viewer: { following: boolean; is_self: boolean };
};
type Activity = { kind: string; created_at: string; post: Post; reply?: Reply };
type Page<T> = { items: T[]; total: number; limit?: number; offset?: number };
const PAGE_SIZE = 20;
const CATEGORIES = [
  { id: "papers", zh: "论文讨论", en: "Papers", icon: BookOpen },
  { id: "implementation", zh: "代码实现", en: "Implementation", icon: Code2 },
  { id: "questions", zh: "问题求助", en: "Questions", icon: HelpCircle },
  {
    id: "career",
    zh: "学术与职业",
    en: "Research & careers",
    icon: BriefcaseBusiness,
  },
  { id: "lounge", zh: "闲聊", en: "Lounge", icon: Coffee },
];
const postHref = (id: string) => `#/community/${encodeURIComponent(id)}`;
const postPath = (id: string) => `community/posts/${encodeURIComponent(id)}`;
function CommunityMarkdown({ body }: { body: string }) {
  return (
    <div className="prose community-prose">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer nofollow">
              {children}
            </a>
          ),
          img: ({ alt }) => (
            <span className="community-image-alt">
              {alt ? `[${alt}]` : "[image]"}
            </span>
          ),
        }}
      >
        {body}
      </Markdown>
    </div>
  );
}
function Category({ value }: { value: string }) {
  const { t } = useI18n();
  const c = CATEGORIES.find((c) => c.id === value);
  return (
    <span className="community-category">{c ? t(c.zh, c.en) : value}</span>
  );
}
function Empty({
  title,
  children,
  icon = <MessagesSquare size={31} />,
}: {
  title: string;
  children?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="community-empty">
      <div className="community-empty-icon">{icon}</div>
      <h3>{title}</h3>
      {children && <div>{children}</div>}
    </div>
  );
}
function Loading() {
  const { t } = useI18n();
  return (
    <div className="social-loading">
      <span className="spinner" />
      {t("正在加载…", "Loading…")}
    </div>
  );
}
function Pagination({
  offset,
  total,
  onChange,
}: {
  offset: number;
  total: number;
  onChange: (offset: number) => void;
}) {
  const { t } = useI18n();
  return total > PAGE_SIZE ? (
    <div className="community-pagination">
      <span>
        {offset + 1}–{Math.min(total, offset + PAGE_SIZE)} / {total}
      </span>
      <button
        disabled={!offset}
        aria-label={t("上一页", "Previous page")}
        onClick={() => onChange(Math.max(0, offset - PAGE_SIZE))}
      >
        <ChevronLeft size={17} />
      </button>
      <button
        disabled={offset + PAGE_SIZE >= total}
        aria-label={t("下一页", "Next page")}
        onClick={() => onChange(offset + PAGE_SIZE)}
      >
        <ChevronRight size={17} />
      </button>
    </div>
  ) : null;
}
function JoinPrompt() {
  const { t } = useI18n();
  return (
    <div className="join-prompt">
      <p>
        {t(
          "使用已验证的机构邮箱加入讨论。",
          "Join the conversation with a verified institutional email.",
        )}
      </p>
      <a className="button" href="#/account">
        {t("登录 / 注册", "Sign in / Join")}
        <ArrowRight size={14} />
      </a>
    </div>
  );
}
function PostCard({ post, activity }: { post: Post; activity?: ReactNode }) {
  return (
    <article className="community-post-card">
      {activity && <div className="activity-label">{activity}</div>}
      <div className="post-card-layout">
        <a
          href={profileHref(post.author)}
          aria-label={post.author.display_name}
        >
          <Avatar user={post.author} size={36} />
        </a>
        <div className="post-card-main">
          <div className="post-card-meta">
            <a className="post-author-name" href={profileHref(post.author)}>
              {post.author.display_name}
            </a>
            <span>·</span>
            <DateLabel value={post.created_at} />
            <Category value={post.category} />
          </div>
          <a className="post-title" href={postHref(post.id)}>
            {post.title}
          </a>
          <p className="post-excerpt">
            {post.body.replace(/[#*`>\[\]]/g, "").slice(0, 190)}
          </p>
          <div className="post-card-counts">
            <span>
              <MessageCircle size={13} />
              {post.counts.replies}
            </span>
            <span>
              <Heart size={13} />
              {post.counts.likes}
            </span>
            <span>
              <Repeat2 size={13} />
              {post.counts.reposts}
            </span>
          </div>
        </div>
      </div>
    </article>
  );
}
function Composer({
  category,
  onClose,
}: {
  category: string;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [title, setTitle] = useState(""),
    [body, setBody] = useState(""),
    [board, setBoard] = useState(category || "papers"),
    [preview, setPreview] = useState(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  async function publish() {
    setBusy(true);
    setError("");
    try {
      const post = await socialApi<Post>("community/posts", {
        title: title.trim(),
        body: body.trim(),
        category: board,
      });
      location.hash = `/community/${encodeURIComponent(post.id)}`;
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="social-panel community-composer">
      <div className="panel-title">
        <h2>{t("发布帖子", "Start a discussion")}</h2>
        <button
          className="icon-button"
          aria-label={t("关闭发帖", "Close composer")}
          onClick={onClose}
          disabled={busy}
        >
          <X size={18} />
        </button>
      </div>
      {error && <SocialNotice error={error} />}
      <form
        className="social-form"
        onSubmit={(e) => {
          e.preventDefault();
          void publish();
        }}
      >
        <label>
          {t("版块", "Category")}
          <select value={board} onChange={(e) => setBoard(e.target.value)}>
            {CATEGORIES.map((c) => (
              <option key={c.id} value={c.id}>
                {t(c.zh, c.en)}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t("标题", "Title")}
          <input
            required
            maxLength={160}
            minLength={3}
            value={title}
            placeholder={t("你想讨论什么？", "What would you like to discuss?")}
            onChange={(e) => setTitle(e.target.value)}
          />
        </label>
        <div className="editor-toggle">
          <button
            className={!preview ? "selected" : ""}
            type="button"
            onClick={() => setPreview(false)}
          >
            {t("编写", "Write")}
          </button>
          <button
            className={preview ? "selected" : ""}
            type="button"
            onClick={() => setPreview(true)}
          >
            {t("预览", "Preview")}
          </button>
          <small>Markdown</small>
        </div>
        {preview ? (
          <div className="compose-preview">
            <CommunityMarkdown
              body={
                body ||
                t("写下内容后在这里预览。", "Your preview will appear here.")
              }
            />
          </div>
        ) : (
          <label className="body-label">
            <span className="sr-only">{t("正文", "Post body")}</span>
            <textarea
              required
              rows={10}
              maxLength={20000}
              value={body}
              placeholder={t(
                "分享背景、思考或代码片段…",
                "Share context, ideas, or a code snippet…",
              )}
              onChange={(e) => setBody(e.target.value)}
            />
          </label>
        )}
        <div className="form-footer">
          <small>
            {t(
              "帖子以你的公开昵称发布。",
              "Posts are published under your public display name.",
            )}
          </small>
          <button className="button" disabled={busy || !body.trim()}>
            {busy ? <span className="spinner" /> : <Plus size={15} />}
            {t("发布", "Publish")}
          </button>
        </div>
      </form>
    </section>
  );
}
export default function Community({ postId }: { postId?: string }) {
  return postId ? <Discussion key={postId} id={postId} /> : <CommunityFeed />;
}
function CommunityFeed() {
  const { t } = useI18n();
  const { user } = useAccount();
  const [category, setCategory] = useState(""),
    [following, setFollowing] = useState(false),
    [query, setQuery] = useState(""),
    [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0),
    [data, setData] = useState<Page<Post>>({ items: [], total: 0 }),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [compose, setCompose] = useState(false),
    [retry, setRetry] = useState(0);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    const params = new URLSearchParams({
      category,
      q: search,
      following: following ? "true" : "false",
      offset: String(offset),
      limit: String(PAGE_SIZE),
    });
    if (following && !user) {
      setData({ items: [], total: 0 });
      setLoading(false);
      return;
    }
    socialApi<Page<Post>>(`community/posts?${params}`)
      .then((data) => {
        if (live) setData(data);
      })
      .catch((e) => {
        if (live) setError(errorMessage(e));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [category, following, search, offset, user?.id, retry]);
  return (
    <main className="social-page community-page">
      <div className="social-page-heading">
        <div>
          <span className="social-eyebrow">PAPERLEET COMMUNITY</span>
          <h1>{t("一起读论文，聊实现。", "Read. Build. Discuss.")}</h1>
          <p>
            {t(
              "分享论文里的想法，也交流代码中的细节。",
              "A place for paper discussions, implementation notes, and questions.",
            )}
          </p>
        </div>
        {user ? (
          <button className="button" onClick={() => setCompose(true)}>
            <Plus size={16} />
            {t("发布帖子", "New post")}
          </button>
        ) : (
          <a className="button" href="#/account">
            <Plus size={16} />
            {t("登录发帖", "Sign in to post")}
          </a>
        )}
      </div>
      <div className="community-layout">
        <aside className="community-sidebar">
          <section className="social-panel board-list">
            <span className="sidebar-label">
              {t("浏览版块", "BROWSE BOARDS")}
            </span>
            <button
              className={!category ? "selected" : ""}
              onClick={() => {
                setCategory("");
                setOffset(0);
              }}
            >
              <MessagesSquare size={17} />
              {t("所有讨论", "All discussions")}
            </button>
            {CATEGORIES.map((c) => (
              <button
                key={c.id}
                className={category === c.id ? "selected" : ""}
                onClick={() => {
                  setCategory(c.id);
                  setOffset(0);
                }}
              >
                <c.icon size={17} />
                {t(c.zh, c.en)}
              </button>
            ))}
          </section>
          <section className="social-panel community-about">
            <BookOpen size={22} className="accent-icon" />
            <h3>{t("从一个问题开始", "Start with a question")}</h3>
            <p>
              {t(
                "说清论文出处、实现思路和遇到的问题，让讨论更容易继续。",
                "Include the paper, your approach, and what you are working through to help others join in.",
              )}
            </p>
            <a href="#">
              {t("浏览论文题库", "Explore the papers")}
              <ArrowRight size={13} />
            </a>
          </section>
        </aside>
        <section className="community-feed">
          {compose && user && (
            <Composer category={category} onClose={() => setCompose(false)} />
          )}
          <div className="community-feed-toolbar">
            <div className="social-tabs compact">
              <button
                className={!following ? "selected" : ""}
                onClick={() => {
                  setFollowing(false);
                  setOffset(0);
                }}
              >
                {t("最新", "Latest")}
              </button>
              <button
                className={following ? "selected" : ""}
                onClick={() => {
                  setFollowing(true);
                  setOffset(0);
                }}
              >
                <Users size={14} />
                {t("关注", "Following")}
              </button>
            </div>
            <form
              className="community-search"
              onSubmit={(e) => {
                e.preventDefault();
                setSearch(query.trim());
                setOffset(0);
              }}
            >
              <Search size={14} />
              <input
                aria-label={t("搜索讨论", "Search discussions")}
                placeholder={t("搜索讨论…", "Search discussions…")}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  if (!e.target.value) {
                    setSearch("");
                    setOffset(0);
                  }
                }}
              />
              <button type="submit" aria-label={t("搜索", "Search")}>
                <ArrowRight size={14} />
              </button>
            </form>
          </div>
          <div className="social-panel feed-items">
            {error ? (
              <div className="feed-error">
                <SocialNotice error={error} />
                <button
                  className="button subtle"
                  onClick={() => setRetry((v) => v + 1)}
                >
                  {t("重试", "Retry")}
                </button>
              </div>
            ) : loading ? (
              <Loading />
            ) : following && !user ? (
              <Empty
                title={t(
                  "关注你感兴趣的人",
                  "Follow people you want to learn with",
                )}
                icon={<Users size={31} />}
              >
                <JoinPrompt />
              </Empty>
            ) : !data.items.length ? (
              <Empty
                title={
                  search
                    ? t("没有找到相关讨论", "No matching discussions")
                    : following
                      ? t(
                          "关注动态会显示在这里",
                          "Your following feed starts here",
                        )
                      : t("这里还没有帖子", "No discussions yet")
                }
              >
                <p>
                  {search
                    ? t("换个关键词试试。", "Try a different search term.")
                    : following
                      ? t(
                          "关注其他研究者，查看他们发布的帖子。",
                          "Follow other members to see their posts here.",
                        )
                      : t(
                          "分享一篇论文或一个实现问题，开启第一个讨论。",
                          "Share a paper or an implementation question to start the first discussion.",
                        )}
                </p>
              </Empty>
            ) : (
              data.items.map((post) => <PostCard key={post.id} post={post} />)
            )}
          </div>
          {!loading && !error && (
            <Pagination
              offset={offset}
              total={data.total}
              onChange={setOffset}
            />
          )}
        </section>
      </div>
    </main>
  );
}
function Discussion({ id }: { id: string }) {
  const { t } = useI18n();
  const { user } = useAccount();
  const [post, setPost] = useState<PostDetail>(),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [body, setBody] = useState(""),
    [target, setTarget] = useState<Reply>(),
    [deleteTarget, setDeleteTarget] = useState<string>();
  async function refresh() {
    const data = await socialApi<PostDetail>(postPath(id));
    setPost(data);
  }
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    socialApi<PostDetail>(postPath(id))
      .then((data) => {
        if (live) setPost(data);
      })
      .catch((e) => {
        if (live) setError(errorMessage(e));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [id, user?.id]);
  async function react(kind: "like" | "repost") {
    if (!user) {
      location.hash = "/account";
      return;
    }
    if (!post) return;
    setBusy(true);
    setError("");
    try {
      const data = await socialApi<Post>(`${postPath(id)}/${kind}`, {
        active: kind === "like" ? !post.viewer.liked : !post.viewer.reposted,
      });
      setPost((prev) => (prev ? { ...data, replies: prev.replies } : prev));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function reply() {
    setBusy(true);
    setError("");
    try {
      await socialApi(`${postPath(id)}/replies`, {
        body: body.trim(),
        ...(target ? { parent_id: target.id } : {}),
      });
      setBody("");
      setTarget(undefined);
      await refresh();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    if (!deleteTarget) return;
    setBusy(true);
    setError("");
    try {
      await socialApi(
        deleteTarget === "post"
          ? `${postPath(id)}/delete`
          : `community/replies/${encodeURIComponent(deleteTarget)}/delete`,
        {},
      );
      if (deleteTarget === "post") location.hash = "/community";
      else await refresh();
      setDeleteTarget(undefined);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="social-page discussion-page">
      <a className="social-back" href="#/community">
        <ArrowLeft size={14} />
        {t("返回社区", "Back to community")}
      </a>
      {error && <SocialNotice error={error} />}
      {loading ? (
        <Loading />
      ) : post ? (
        <>
          <article className="social-panel discussion-post">
            <div className="discussion-heading">
              <Category value={post.category} />
              <h1>{post.title}</h1>
              <div className="discussion-author">
                <PersonLink user={post.author} />
                <InstitutionBadge institution={post.author.institution} />
                <DateLabel value={post.created_at} />
                {post.viewer.can_delete && (
                  <button
                    className="icon-button delete-action"
                    aria-label={t("删除帖子", "Delete post")}
                    onClick={() => setDeleteTarget("post")}
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            </div>
            <CommunityMarkdown body={post.body} />
            <div className="discussion-actions">
              <button
                className={post.viewer.liked ? "active" : ""}
                disabled={busy}
                aria-pressed={post.viewer.liked}
                onClick={() => void react("like")}
              >
                <Heart
                  size={16}
                  fill={post.viewer.liked ? "currentColor" : "none"}
                />
                {t("点赞", "Like")}
                <span>{post.counts.likes}</span>
              </button>
              <button
                className={post.viewer.reposted ? "active" : ""}
                disabled={busy}
                aria-pressed={post.viewer.reposted}
                onClick={() => void react("repost")}
              >
                <Repeat2 size={16} />
                {post.viewer.reposted
                  ? t("已转发", "Reposted")
                  : t("转发", "Repost")}
                <span>{post.counts.reposts}</span>
              </button>
              <span>
                <MessageCircle size={15} />
                {post.counts.replies} {t("条回复", "replies")}
              </span>
            </div>
          </article>
          <section className="social-panel discussion-replies">
            <div className="panel-title">
              <h2>
                {t("回复", "Replies")} <span>{post.counts.replies}</span>
              </h2>
            </div>
            {post.replies.length ? (
              post.replies.map((r) => (
                <article
                  key={r.id}
                  className={`community-reply ${r.parent_id ? "nested-reply" : ""}`}
                  id={`reply-${r.id}`}
                >
                  <div className="reply-heading">
                    <PersonLink user={r.author} compact />
                    <DateLabel value={r.created_at} />
                    {r.parent_id && (
                      <button
                        className="reply-context"
                        onClick={() =>
                          document
                            .getElementById(`reply-${r.parent_id}`)
                            ?.scrollIntoView({
                              behavior: "smooth",
                              block: "center",
                            })
                        }
                      >
                        {t("回复", "Replying to")}{" "}
                        {post.replies.find(
                          (parent) => parent.id === r.parent_id,
                        )?.author.display_name ||
                          t("已删除的回复", "a deleted reply")}
                      </button>
                    )}
                    {r.can_delete && (
                      <button
                        className="icon-button delete-action"
                        aria-label={t("删除回复", "Delete reply")}
                        onClick={() => setDeleteTarget(r.id)}
                      >
                        <Trash2 size={13} />
                      </button>
                    )}
                  </div>
                  <CommunityMarkdown body={r.body} />
                  {user && (
                    <button
                      className="reply-link"
                      onClick={() => {
                        setTarget(r);
                        document.getElementById("reply-body")?.focus();
                      }}
                    >
                      <MessageCircle size={12} />
                      {t("回复", "Reply")}
                    </button>
                  )}
                </article>
              ))
            ) : (
              <p className="no-replies">
                {t(
                  "还没有回复，分享你的想法吧。",
                  "No replies yet. Share your thoughts.",
                )}
              </p>
            )}
          </section>
          {user ? (
            <section className="social-panel reply-composer">
              <div className="panel-title">
                <h3>{t("参与讨论", "Join the discussion")}</h3>
                <PersonLink user={user} compact />
              </div>
              {target && (
                <div className="reply-target">
                  <span>
                    {t("回复", "Replying to")} {target.author.display_name}
                  </span>
                  <button
                    className="icon-button"
                    aria-label={t("取消指定回复", "Cancel reply target")}
                    onClick={() => setTarget(undefined)}
                  >
                    <X size={14} />
                  </button>
                </div>
              )}
              <form
                className="social-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  void reply();
                }}
              >
                <label>
                  <span className="sr-only">{t("回复内容", "Your reply")}</span>
                  <textarea
                    id="reply-body"
                    required
                    rows={5}
                    maxLength={10000}
                    value={body}
                    placeholder={t(
                      "写下你的回复…（支持 Markdown）",
                      "Write a reply… (Markdown supported)",
                    )}
                    onChange={(e) => setBody(e.target.value)}
                  />
                </label>
                <div className="form-footer">
                  <small>
                    {t(
                      "回复会公开展示在你的个人主页。",
                      "Replies appear publicly on your profile.",
                    )}
                  </small>
                  <button className="button" disabled={busy || !body.trim()}>
                    <MessageCircle size={14} />
                    {t("发布回复", "Post reply")}
                  </button>
                </div>
              </form>
            </section>
          ) : (
            <section className="social-panel">
              <JoinPrompt />
            </section>
          )}
          {deleteTarget && (
            <div className="delete-confirm" role="alert">
              <span>
                {t(
                  "确认删除？此操作无法撤销。",
                  "Delete this content? This cannot be undone.",
                )}
              </span>
              <button
                className="button subtle"
                disabled={busy}
                onClick={() => setDeleteTarget(undefined)}
              >
                {t("取消", "Cancel")}
              </button>
              <button
                className="button danger"
                disabled={busy}
                onClick={() => void remove()}
              >
                {t("删除", "Delete")}
              </button>
            </div>
          )}
        </>
      ) : !error ? (
        <Empty title={t("帖子不存在", "Post not found")} />
      ) : null}
    </main>
  );
}
export function UserProfile({ username }: { username: string }) {
  const { t } = useI18n();
  const { user } = useAccount();
  const [profile, setProfile] = useState<Profile>(),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [tab, setTab] = useState("posts"),
    [offset, setOffset] = useState(0),
    [activity, setActivity] = useState<Page<Activity>>({ items: [], total: 0 }),
    [connections, setConnections] = useState<Page<Connection>>({
      items: [],
      total: 0,
    }),
    [activityLoading, setActivityLoading] = useState(true),
    [activityError, setActivityError] = useState(""),
    [busy, setBusy] = useState(false),
    [revision, setRevision] = useState(0);
  const path = `community/profiles/${encodeURIComponent(username)}`;
  const connectionTab = tab === "following" || tab === "followers";
  useEffect(() => {
    setProfile(undefined);
    setTab("posts");
    setOffset(0);
  }, [username]);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    socialApi<Profile>(path)
      .then((data) => {
        if (live) setProfile(data);
      })
      .catch((e) => {
        if (live) {
          setProfile(undefined);
          setError(errorMessage(e));
        }
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [username, user?.id, revision]);
  useEffect(() => {
    let live = true;
    setActivityLoading(true);
    setActivityError("");
    const params = new URLSearchParams({
      offset: String(offset),
      limit: String(PAGE_SIZE),
      ...(connectionTab ? { kind: tab } : { tab }),
    });
    const promise = connectionTab
      ? socialApi<Page<Connection>>(`${path}/connections?${params}`).then(
          (data) => {
            if (live) setConnections(data);
          },
        )
      : socialApi<Page<Activity>>(`${path}/activity?${params}`).then((data) => {
          if (live) setActivity(data);
        });
    promise
      .catch((e) => {
        if (live) {
          setActivity({ items: [], total: 0 });
          setConnections({ items: [], total: 0 });
          setActivityError(errorMessage(e));
        }
      })
      .finally(() => {
        if (live) setActivityLoading(false);
      });
    return () => {
      live = false;
    };
  }, [username, tab, offset, user?.id, revision]);
  async function follow(target: string, active: boolean) {
    if (!user) {
      location.hash = "/account";
      return;
    }
    setBusy(true);
    setError("");
    try {
      const updated = await socialApi<Profile>(
        `community/follow/${encodeURIComponent(target)}`,
        { active },
      );
      if (target === username) setProfile(updated);
      setRevision((v) => v + 1);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  const tabs = [
    ["posts", t("帖子", "Posts")],
    ["replies", t("回复", "Replies")],
    ["likes", t("点赞", "Likes")],
    ["reposts", t("转发", "Reposts")],
    ["following", t("关注", "Following")],
    ["followers", t("粉丝", "Followers")],
  ] as const;
  const chooseTab = (value: string) => {
    setTab(value);
    setOffset(0);
  };
  return (
    <main className="social-page profile-page">
      <a className="social-back" href="#/community">
        <ArrowLeft size={14} />
        {t("社区", "Community")}
      </a>
      {error && <SocialNotice error={error} />}
      {loading && !profile ? (
        <Loading />
      ) : profile ? (
        <>
          <section className="social-panel profile-header">
            <div className="profile-cover">
              <span>PaperLeet</span>
            </div>
            <div className="profile-header-inner">
              <Avatar user={profile} size={84} />
              <div className="profile-primary">
                <h1>{profile.display_name}</h1>
                <span className="profile-handle">@{profile.username}</span>
                <div className="profile-facts">
                  <InstitutionBadge institution={profile.institution} />
                  {profile.country && (
                    <span>
                      <Globe2 size={13} />
                      {profile.country}
                    </span>
                  )}
                  <span>
                    {t("加入于", "Joined")}{" "}
                    <DateLabel value={profile.created_at} />
                  </span>
                </div>
                {profile.bio && <p className="profile-bio">{profile.bio}</p>}
              </div>
              {profile.viewer.is_self ? (
                <a className="button subtle profile-edit" href="#/account">
                  <Settings2 size={14} />
                  {t("编辑资料", "Edit profile")}
                </a>
              ) : (
                <button
                  className={`button ${profile.viewer.following ? "subtle" : ""} profile-edit`}
                  disabled={busy}
                  aria-label={
                    profile.viewer.following
                      ? t("取消关注", "Unfollow")
                      : t("关注", "Follow")
                  }
                  onClick={() =>
                    void follow(username, !profile.viewer.following)
                  }
                >
                  {profile.viewer.following ? (
                    <Check size={14} />
                  ) : (
                    <UserPlus size={14} />
                  )}
                  {profile.viewer.following
                    ? t("已关注", "Following")
                    : t("关注", "Follow")}
                </button>
              )}
            </div>
            <div className="profile-stats">
              <button onClick={() => chooseTab("posts")}>
                <b>{profile.counts.posts}</b>
                {t("帖子", "Posts")}
              </button>
              <button onClick={() => chooseTab("replies")}>
                <b>{profile.counts.replies}</b>
                {t("回复", "Replies")}
              </button>
              <button onClick={() => chooseTab("following")}>
                <b>{profile.counts.following}</b>
                {t("关注", "Following")}
              </button>
              <button onClick={() => chooseTab("followers")}>
                <b>{profile.counts.followers}</b>
                {t("粉丝", "Followers")}
              </button>
            </div>
          </section>
          <div className="social-tabs profile-tabs">
            {tabs.map(([key, label]) => (
              <button
                key={key}
                className={tab === key ? "selected" : ""}
                onClick={() => chooseTab(key)}
              >
                {label}
                <span>{profile.counts[key]}</span>
              </button>
            ))}
          </div>
          <section className="social-panel profile-activity">
            {activityLoading ? (
              <Loading />
            ) : activityError ? (
              <div className="feed-error">
                <SocialNotice error={activityError} />
                <button
                  className="button subtle"
                  onClick={() => setRevision((v) => v + 1)}
                >
                  {t("重试", "Retry")}
                </button>
              </div>
            ) : connectionTab ? (
              connections.items.length ? (
                connections.items.map((member) => (
                  <div className="connection-card" key={member.id}>
                    <a href={profileHref(member)}>
                      <Avatar user={member} size={45} />
                    </a>
                    <div>
                      <a href={profileHref(member)}>
                        <b>{member.display_name}</b>
                        <span>@{member.username}</span>
                      </a>
                      <InstitutionBadge institution={member.institution} />
                      {member.bio && <p>{member.bio}</p>}
                    </div>
                    {!member.viewer.is_self && (
                      <button
                        className={`button ${member.viewer.following ? "subtle" : ""}`}
                        disabled={busy}
                        aria-label={
                          member.viewer.following
                            ? t("取消关注", "Unfollow")
                            : t("关注", "Follow")
                        }
                        onClick={() =>
                          void follow(member.username, !member.viewer.following)
                        }
                      >
                        {member.viewer.following
                          ? t("已关注", "Following")
                          : t("关注", "Follow")}
                      </button>
                    )}
                  </div>
                ))
              ) : (
                <Empty
                  title={
                    tab === "following"
                      ? t("还没有关注的人", "No following yet")
                      : t("还没有粉丝", "No followers yet")
                  }
                  icon={<Users size={30} />}
                />
              )
            ) : activity.items.length ? (
              activity.items.map((item, index) => (
                <div
                  key={`${item.kind}-${item.reply?.id || item.post.id}-${index}`}
                >
                  {item.reply ? (
                    <article className="profile-reply-card">
                      <div className="activity-label">
                        <MessageCircle size={13} />
                        {t("回复了", "Replied to")}
                        <a href={postHref(item.post.id)}>{item.post.title}</a>
                        <DateLabel value={item.created_at} />
                      </div>
                      <CommunityMarkdown body={item.reply.body} />
                    </article>
                  ) : (
                    <PostCard
                      post={item.post}
                      activity={
                        tab === "likes" ? (
                          <>
                            <Heart size={13} />
                            {t("点赞了这个帖子", "Liked this post")}
                            <DateLabel value={item.created_at} />
                          </>
                        ) : tab === "reposts" ? (
                          <>
                            <Repeat2 size={13} />
                            {t("转发了这个帖子", "Reposted this post")}
                            <DateLabel value={item.created_at} />
                          </>
                        ) : undefined
                      }
                    />
                  )}
                </div>
              ))
            ) : (
              <Empty title={t("还没有相关动态", "No activity here yet")}>
                <p>
                  {t(
                    "发布、回复或互动后，记录会显示在这里。",
                    "Posts and interactions will appear here as they happen.",
                  )}
                </p>
              </Empty>
            )}
          </section>
          <Pagination
            offset={offset}
            total={connectionTab ? connections.total : activity.total}
            onChange={setOffset}
          />
        </>
      ) : null}
    </main>
  );
}
