import assert from "node:assert/strict";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";
const server = await createServer({
  server: { middlewareMode: true },
  appType: "custom",
});
try {
  const { TestSignIn, SignIn, ProfileSettings } =
    await server.ssrLoadModule("/src/Account.tsx");
  const { PersonLink } = await server.ssrLoadModule("/src/social.tsx");
  const login = renderToString(
    React.createElement(TestSignIn, { enabled: true }),
  );
  assert(
    login.includes('type="password"'),
    "Access codes must use a masked input",
  );
  assert(
    login.includes('autoComplete="username"'),
    "Test login must request the issued account name",
  );
  assert(
    !login.includes('type="email"'),
    "Test mode must not require an institutional email",
  );
  assert(
    !login.includes("one-time-code"),
    "Test mode must not request an emailed OTP",
  );
  assert(
    !login.includes("申请收录"),
    "Institution review is not part of test login",
  );
  const institutional = renderToString(
    React.createElement(SignIn, { enabled: true }),
  );
  assert(
    institutional.includes('type="email"'),
    "Institutional sign-in must remain available in its own mode",
  );
  assert(
    institutional.includes("发送验证码"),
    "Institutional sign-in must keep email verification",
  );
  const testUser = {
    id: "qa-test",
    username: "public_handle",
    login_username: "issued_login",
    display_name: "QA Test User",
    avatar_color: "#4f7c73",
    bio: "",
    created_at: "2026-10-13T00:00:00Z",
    is_test_user: true,
    email: "",
    role: "member",
    institution: null,
    institution_verified: false,
    country: "",
    show_institution: false,
    show_country: false,
    show_progress: false,
  };
  const settings = renderToString(
    React.createElement(ProfileSettings, { user: testUser }),
  );
  assert(
    settings.includes("issued_login"),
    "Settings must display the stable login username",
  );
  assert(
    settings.includes("public_handle"),
    "The public handle must remain distinct from login credentials",
  );
  assert(
    !settings.includes("已验证的账户"),
    "Test accounts must not claim verified institutional identity",
  );
  assert(
    !settings.includes("显示已验证机构"),
    "A null test institution must not have a visibility control",
  );
  const publicLink = renderToString(
    React.createElement(PersonLink, { user: testUser }),
  );
  assert(
    publicLink.includes("内测账号"),
    "Public test identities must be labeled",
  );
  assert(
    !publicLink.includes("issued_login"),
    "The private login name must not appear in public identity links",
  );
  console.log(
    "PASS test-account login, masked access codes, separate public identity and no institutional verification claims",
  );
} finally {
  await server.close();
}
