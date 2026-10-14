import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { calendarCells } = await server.ssrLoadModule("/src/profile-progress.ts");
  const accepted = { date: "2024-02-29", submissions: 4, attempted: 2, solved: 1, newly_solved: 1 };
  const leap = calendarCells("2024-01-01", "2024-12-31", [accepted]);
  assert.equal(leap.filter((cell) => cell.inRange).length, 366, "Leap day must be included");
  assert.equal(new Set(leap.map((cell) => cell.date)).size, leap.length, "Each date must occupy exactly one cell");
  assert.equal(leap.length % 7, 0, "The calendar must use complete weeks");
  assert.equal(new Date(`${leap[0].date}T00:00:00Z`).getUTCDay(), 0, "Calendar columns start on Sunday");
  assert.deepEqual(leap.find((cell) => cell.date === accepted.date).day, accepted, "Repeated submissions and unique problems must remain distinct");
  assert.equal(leap.find((cell) => cell.date === "2024-03-01").day.submissions, 0, "Inactive dates must not inherit adjacent activity");
  assert.equal(leap[0].inRange, false, "Padding in another year must not become an interactive day");

  for (const timezone of ["America/New_York", "Asia/Shanghai", "Pacific/Auckland"]) {
    process.env.TZ = timezone;
    const cells = calendarCells("2026-03-07", "2026-03-10", []);
    assert.deepEqual(cells.filter((cell) => cell.inRange).map((cell) => cell.date),
      ["2026-03-07", "2026-03-08", "2026-03-09", "2026-03-10"],
      `Date-only cells must not skip or repeat at DST boundaries in ${timezone}`);
  }
  const firstDay = calendarCells("2026-01-01", "2026-01-01", []);
  assert.equal(firstDay.filter((cell) => cell.inRange).length, 1, "A new-year calendar must allow only the current date");
  assert.deepEqual(calendarCells("2026-02-01", "2026-01-01", []), [], "Invalid ranges must not generate activity");
  console.log("PASS profile calendar leap years, week alignment, daily dedup fields, date bounds and DST independence");
} finally {
  await server.close();
}
