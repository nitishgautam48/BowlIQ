import { test } from "node:test";
import assert from "node:assert/strict";
import { frameToTime, isSlowmo, timeToFrame } from "../lib/frames.ts";
import { fmtVal, fmtNum } from "../lib/format.ts";

const video = { fps: 120, width: 1280, height: 720, frame_count: 150, duration_s: 1.25, slowmo: { start_frame: 68, end_frame: 140, factor: 4 } };

test("frame → time → frame round-trips for every frame (slow-motion window repeats frames 4×)", () => {
  for (let f = 0; f < video.frame_count; f++) {
    assert.equal(timeToFrame(frameToTime(f, video), video), f, `frame ${f}`);
  }
});

test("slow-motion window detection", () => {
  assert.equal(isSlowmo(67, video), false);
  assert.equal(isSlowmo(68, video), true);
  assert.equal(isSlowmo(140, video), true);
  assert.equal(isSlowmo(141, video), false);
});

test("time inside a repeated slow-motion frame maps to that frame", () => {
  const t0 = (68 + 0.1) / 120; // first display tick of frame 68
  const t3 = (68 + 3.9) / 120; // last display tick of frame 68
  assert.equal(timeToFrame(t0, video), 68);
  assert.equal(timeToFrame(t3, video), 68);
  assert.equal(timeToFrame((68 + 4.1) / 120, video), 69);
});

test("timeToFrame clamps to the clip", () => {
  assert.equal(timeToFrame(-1, video), 0);
  assert.equal(timeToFrame(999, video), video.frame_count - 1);
});

test("value formatting", () => {
  assert.equal(fmtVal(17.4, "°"), "17°");
  assert.equal(fmtVal(81.33, "% height"), "81.3 % height");
  assert.equal(fmtVal(null, "m/s"), "—");
  assert.equal(fmtNum(-5.21, "m/s"), "−5.2");
});
