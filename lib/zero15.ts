import type { Candle } from './binance'
import { atrSeries } from './indicators'
import { impulseMacd, supertrend, type Side, type TradeOutcome } from './impulse5m'

// ---------------------------------------------------------------------------
// MTF Impulse MACD + Supertrend engine: 15M = trend, 5M = entry, zero-line dead-zone filter. Pure logic, no I/O.
//
//   15M (last fully CLOSED candle):  Impulse MACD line > signal, both lines above zero, histogram positive, Supertrend 12,4 bullish
//   5M  (candle close):              Impulse MACD crosses above its signal, both lines above zero, histogram positive, Supertrend bullish
//   Zero-line filter:                every line must sit further from zero than DEAD_ZONE x ATR (ATR of the same timeframe, so the
//                                    buffer scales with each coin) and both lines must be on the same side of zero; the histogram
//                                    must be at least MIN_HIST x ATR. SELL is the exact mirror image.
//
// Every 5M signal is evaluated "as of" that candle's close: the 15M timeframe contributes only candles that had already closed by
// then, so nothing repaints and nothing looks ahead. One signal per 5M crossover, plus a per-coin cooldown.
// ---------------------------------------------------------------------------

export const Z15_VERSION = 'z15-1.0'
const MS5 = 300_000
const MS15 = 900_000
const WARM = 80

export interface Z15Config {
  /** LazyBear length (smma of high / low, zlema of close). */
  length: number
  signal: number
  stPeriod: number
  stMult: number
  /** Dead zone around zero, in ATRs of the same timeframe. Lines closer than this to zero = MACD near zero. */
  deadZone: number
  /** Minimum |histogram| in ATRs. */
  minHist: number
  /** Stop = this many 5M ATR from the entry. */
  atrSl: number
  /** Take profit = this many R. */
  rr: number
  /** Stops wider than this % of price are rejected. */
  maxRiskPct: number
  /** 5M bars during which the same coin cannot signal again. */
  cooldownBars: number
  /** A signal counts as fresh (shown / popped up) for this many 5M bars after its candle. */
  freshBars: number
  feeBps: number
  slipBps: number
  /** Risk per trade, % of the account balance (used for the position-size hint only; nothing is traded automatically). */
  riskPct: number
}

export const DEFAULT_Z15: Z15Config = {
  length: 34,
  signal: 9,
  stPeriod: 12,
  stMult: 4,
  deadZone: 0.15,
  minHist: 0.03,
  atrSl: 1.5,
  rr: 2,
  maxRiskPct: 3,
  cooldownBars: 3,
  freshBars: 2,
  feeBps: 4,
  slipBps: 2,
  riskPct: 1,
}

export function normalizeZ15(p: Partial<Record<keyof Z15Config, unknown>> = {}): Z15Config {
  const d = DEFAULT_Z15
  const num = (k: keyof Z15Config, lo: number, hi: number, int = false) => {
    const x = Number(p[k] ?? d[k])
    const v = Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : d[k]
    return int ? Math.round(v) : v
  }
  return {
    length: num('length', 5, 100, true),
    signal: num('signal', 2, 50, true),
    stPeriod: num('stPeriod', 2, 50, true),
    stMult: num('stMult', 0.5, 10),
    deadZone: num('deadZone', 0, 3),
    minHist: num('minHist', 0, 2),
    atrSl: num('atrSl', 0.5, 6),
    rr: num('rr', 1, 5),
    maxRiskPct: num('maxRiskPct', 0.2, 10),
    cooldownBars: num('cooldownBars', 0, 60, true),
    freshBars: num('freshBars', 1, 6, true),
    feeBps: num('feeBps', 0, 30),
    slipBps: num('slipBps', 0, 30),
    riskPct: num('riskPct', 0.1, 5),
  }
}

export type Dir = 'BULLISH' | 'BEARISH' | 'NEUTRAL'

interface S {
  c: Candle[]
  ms: number
  ml: number[]
  sg: number[]
  hist: number[]
  stDir: number[]
  atr: number[]
}

function series(c: Candle[], ms: number, cfg: Z15Config): S {
  const m = impulseMacd(c, cfg.length, cfg.signal)
  const st = supertrend(c, cfg.stPeriod, cfg.stMult)
  return { c, ms, ml: m.md, sg: m.signal, hist: m.hist, stDir: st.dir, atr: atrSeries(c, 14) }
}

const dirOf = (s: S, i: number): Dir => (s.ml[i] > s.sg[i] ? 'BULLISH' : s.ml[i] < s.sg[i] ? 'BEARISH' : 'NEUTRAL')
const crossOf = (s: S, i: number): Side | null =>
  i < 1 ? null : s.ml[i] > s.sg[i] && s.ml[i - 1] <= s.sg[i - 1] ? 'BUY' : s.ml[i] < s.sg[i] && s.ml[i - 1] >= s.sg[i - 1] ? 'SELL' : null

export interface Z15Check {
  key: 't15' | 'm5' | 'st' | 'zero' | 'mom' | 'sl' | 'cooldown'
  label: string
  ok: boolean
  detail: string
  /** Exact line shown when this is the first failing check. */
  wait: string
}

export interface Z15Part {
  label: string
  pts: number
  max: number
}

export interface Z15Plan {
  entry: number
  sl: number
  tp: number
  risk: number
  riskPct: number
  rr: number
  netRR: number
}

export interface Z15Signal extends Z15Plan {
  id: string
  symbol: string
  side: Side
  /** Close time of the 5M signal candle (ms). */
  time: number
  barsAgo: number
  m15: { line: number; signal: number; hist: number }
  m5: { line: number; signal: number; hist: number }
  stBull15: boolean
  stBull5: boolean
  histExpanding: boolean
  score: number
  parts: Z15Part[]
  conditions: string[]
  version: string
  outcome: TradeOutcome
}

export interface Z15Rejection {
  id: string
  symbol: string
  side: Side
  time: number
  reason: string
  failed: string[]
}

export type ZeroState = 'CLEAR' | 'NEAR ZERO' | 'OPPOSITE SIDES'

export interface Z15Row {
  symbol: string
  price: number
  barTime: number
  m15: { dir: Dir; line: number; signal: number; hist: number }
  m5: { dir: Dir; line: number; signal: number; hist: number }
  st15: 'BULLISH' | 'BEARISH'
  st5: 'BULLISH' | 'BEARISH'
  zero: ZeroState
  agree: boolean
  stAgree: boolean
  status: 'BUY' | 'SELL' | 'WAIT'
  /** Exact reason while WAIT, or a one-line confirmation. */
  text: string
  checks: Z15Check[]
  signal: Z15Signal | null
  score: number | null
  rejections: Z15Rejection[]
  version: string
}

export interface Z15Run {
  signals: Z15Signal[]
  rejections: Z15Rejection[]
  row: Z15Row
}

const W = (s: string) => `WAIT / NO TRADE — ${s}`
export const Z15_REASONS = {
  t15: W('15M trend disagreement'),
  m5: W('5M MACD not confirmed'),
  st: W('Supertrend disagreement'),
  zero: W('MACD near zero'),
  mom: W('Insufficient momentum'),
  sl: W('Invalid stop loss'),
  cooldown: W('Cooldown active'),
} as const

/** Position-size hint: units to buy so a stop-out loses `riskPct` % of the balance. Never executes anything. */
export function positionSize(balance: number, riskPct: number, riskDistance: number, entry: number) {
  if (!(balance > 0) || !(riskDistance > 0)) return null
  const qty = (balance * (riskPct / 100)) / riskDistance
  return { qty, notional: qty * entry, riskAmount: balance * (riskPct / 100) }
}

/** How the zero-line looks for the two timeframes together (independent of the trade side). */
function zeroState(s5: S, i5: number, s15: S, i15: number, cfg: Z15Config): ZeroState {
  const sides = [Math.sign(s5.ml[i5]) * Math.sign(s5.sg[i5]), Math.sign(s15.ml[i15]) * Math.sign(s15.sg[i15])]
  if (sides.some((x) => x < 0) || Math.sign(s5.ml[i5]) * Math.sign(s15.ml[i15]) < 0) return 'OPPOSITE SIDES'
  const near = (s: S, i: number) => Math.min(Math.abs(s.ml[i]), Math.abs(s.sg[i])) < cfg.deadZone * s.atr[i]
  return near(s5, i5) || near(s15, i15) ? 'NEAR ZERO' : 'CLEAR'
}

export function runZ15(symbol: string, c5: Candle[], c15: Candle[], cfgIn: Partial<Z15Config> = {}): Z15Run | null {
  const cfg = normalizeZ15(cfgIn)
  if (c5.length < WARM + 5 || c15.length < WARM + 5) return null
  const s5 = series(c5, MS5, cfg)
  const s15 = series(c15, MS15, cfg)
  const n = c5.length
  const last = n - 1
  const stOk = (s: S, i: number, bull: boolean) => (bull ? s.stDir[i] === 1 : s.stDir[i] === -1)

  /** Evaluate every rule for `side` at 5M bar i with the 15M pointer p. `crossOk` says whether the 5M crossover is on this bar. */
  const assess = (i: number, p: number, bull: boolean, crossOk: boolean, sinceLast: number) => {
    const checks: Z15Check[] = []
    const add = (key: Z15Check['key'], label: string, ok: boolean, detail: string) => checks.push({ key, label, ok, detail, wait: Z15_REASONS[key] })
    const w = (b: boolean) => (b ? 'bullish' : 'bearish')
    const l15 = s15.ml[p]
    const g15 = s15.sg[p]
    const h15 = s15.hist[p]
    const l5 = s5.ml[i]
    const g5 = s5.sg[i]
    const h5 = s5.hist[i]
    const side15 = bull ? l15 > g15 && l15 > 0 && g15 > 0 && h15 > 0 : l15 < g15 && l15 < 0 && g15 < 0 && h15 < 0
    add('t15', '15M trend (closed candle)', side15, side15 ? `15M Impulse MACD ${w(bull)}, both lines ${bull ? 'above' : 'below'} zero, histogram ${bull ? 'positive' : 'negative'}` : `15M line ${l15.toPrecision(4)} / signal ${g15.toPrecision(4)} / histogram ${h15.toPrecision(4)} do not all point ${w(bull)}`)
    add('m5', '5M crossover on the candle close', crossOk, crossOk ? `5M line crossed ${bull ? 'above' : 'below'} its signal on this candle` : 'no fresh 5M crossover in this direction')
    const st15 = stOk(s15, p, bull)
    const st5 = stOk(s5, i, bull)
    add('st', 'Supertrend 12,4 on 15M and 5M', st15 && st5, st15 && st5 ? `15M and 5M Supertrend ${w(bull)}` : `${!st15 ? '15M' : ''}${!st15 && !st5 ? ' and ' : ''}${!st5 ? '5M' : ''} Supertrend is ${w(!bull)}`)
    const side5 = bull ? l5 > 0 && g5 > 0 : l5 < 0 && g5 < 0
    const dz5 = Math.min(Math.abs(l5), Math.abs(g5)) / s5.atr[i]
    const dz15 = Math.min(Math.abs(l15), Math.abs(g15)) / s15.atr[p]
    const zeroOk = side5 && side15 && dz5 >= cfg.deadZone && dz15 >= cfg.deadZone
    add('zero', 'Zero-line dead zone', zeroOk, zeroOk ? `lines clear of zero (5M ${dz5.toFixed(2)} ATR, 15M ${dz15.toFixed(2)} ATR; limit ${cfg.deadZone})` : !side5 || !side15 ? 'lines are on opposite sides of zero or on the wrong side' : `lines inside the dead zone (5M ${dz5.toFixed(2)} ATR, 15M ${dz15.toFixed(2)} ATR; limit ${cfg.deadZone})`)
    const m5 = Math.abs(h5) / s5.atr[i]
    const m15 = Math.abs(h15) / s15.atr[p]
    const momOk = m5 >= cfg.minHist && m15 >= cfg.minHist
    add('mom', 'Histogram momentum', momOk, `histogram ${m5.toFixed(2)} ATR (5M) / ${m15.toFixed(2)} ATR (15M), minimum ${cfg.minHist}`)

    const entry = s5.c[i].close
    const risk = cfg.atrSl * s5.atr[i]
    const riskPct = (risk / entry) * 100
    const slOk = Number.isFinite(risk) && risk > 0 && riskPct <= cfg.maxRiskPct
    add('sl', 'ATR stop loss', slOk, slOk ? `stop ${cfg.atrSl} × 5M ATR = ${riskPct.toFixed(2)}% of price` : `stop would be ${riskPct.toFixed(2)}% of price (limit ${cfg.maxRiskPct}%)`)
    add('cooldown', `Coin cooldown (${cfg.cooldownBars} × 5M bars)`, sinceLast >= cfg.cooldownBars, sinceLast >= cfg.cooldownBars ? 'not cooling down' : `last signal ${sinceLast} bar${sinceLast === 1 ? '' : 's'} ago`)

    const costR = ((2 * (cfg.feeBps + cfg.slipBps)) / 10_000) * (entry / risk)
    const plan: Z15Plan = {
      entry,
      sl: bull ? entry - risk : entry + risk,
      tp: bull ? entry + cfg.rr * risk : entry - cfg.rr * risk,
      risk,
      riskPct,
      rr: cfg.rr,
      netRR: (cfg.rr - costR) / (1 + costR),
    }
    // quality score 0-100 (setup quality, NOT a win probability): ten documented parts
    const sc = (v: number, full: number, max: number) => Math.round(max * Math.min(1, Math.max(0, v / full)))
    const parts: Z15Part[] = [
      { label: '15M histogram strength', pts: sc(m15, 0.3, 15), max: 15 },
      { label: '5M histogram strength', pts: sc(m5, 0.3, 15), max: 15 },
      { label: '5M histogram expanding', pts: (bull ? h5 > s5.hist[i - 1] : h5 < s5.hist[i - 1]) ? 10 : 0, max: 10 },
      { label: '15M histogram expanding', pts: (bull ? h15 > s15.hist[p - 1] : h15 < s15.hist[p - 1]) ? 5 : 0, max: 5 },
      { label: '15M distance from zero', pts: sc(dz15, 1, 10), max: 10 },
      { label: '5M distance from zero', pts: sc(dz5, 1, 10), max: 10 },
      { label: 'Supertrend agreement (15M + 5M)', pts: (st15 ? 7 : 0) + (st5 ? 8 : 0), max: 15 },
      { label: 'Both timeframes agree', pts: side15 && side5 ? 10 : 0, max: 10 },
    ]
    // last 10 points: a calm signal candle (not oversized)
    const range = s5.c[i].high - s5.c[i].low
    parts.push({ label: 'Signal candle not oversized', pts: range <= 2.5 * s5.atr[i] ? 10 : 0, max: 10 })
    const score = parts.reduce((a, x) => a + x.pts, 0)
    return { checks, plan, score, parts, st15, st5, expanding: parts[2].pts > 0 }
  }

  const signals: Z15Signal[] = []
  const rejections: Z15Rejection[] = []
  let lastIdx = -1e9
  let p15Last = -1
  const ptrAt: number[] = new Array(n).fill(-1)
  let p = -1
  for (let i = 0; i < n; i++) {
    const T = c5[i].openTime + MS5
    while (p + 1 < c15.length && c15[p + 1].openTime + MS15 <= T) p++
    ptrAt[i] = p
    if (i === last) p15Last = p
    if (i < WARM || p < WARM) continue
    const cr = crossOf(s5, i)
    if (!cr) continue
    const bull = cr === 'BUY'
    const a = assess(i, p, bull, true, i - lastIdx)
    const id = `${symbol}-z15-${cr}-${T}`
    const failed = a.checks.filter((c) => !c.ok)
    if (failed.length) {
      rejections.push({ id, symbol, side: cr, time: T, reason: failed[0].wait, failed: failed.map((f) => f.label) })
      continue
    }
    const entry = a.plan.entry
    let outcome: TradeOutcome = { result: 'OPEN', r: 0, pnlPct: 0, closedAt: null }
    const cost = ((2 * (cfg.feeBps + cfg.slipBps)) / 10_000) * (entry / a.plan.risk)
    let done = false
    for (let j = i + 1; j < n && !done; j++) {
      const c = c5[j]
      const hitSl = bull ? c.low <= a.plan.sl : c.high >= a.plan.sl
      const hitTp = bull ? c.high >= a.plan.tp : c.low <= a.plan.tp
      if (hitSl || hitTp) {
        // both inside one candle: the order is unknowable, so the stop is assumed first
        const r = (hitSl ? -1 : cfg.rr) - cost
        outcome = { result: hitSl ? 'LOSS' : 'WIN', r, pnlPct: ((r * a.plan.risk) / entry) * 100, closedAt: c.openTime + MS5 }
        done = true
      }
    }
    if (!done) {
      const r = (bull ? c5[last].close - entry : entry - c5[last].close) / a.plan.risk - cost
      outcome = { result: 'OPEN', r, pnlPct: ((r * a.plan.risk) / entry) * 100, closedAt: null }
    }
    signals.push({
      ...a.plan,
      id,
      symbol,
      side: cr,
      time: T,
      barsAgo: last - i,
      m15: { line: s15.ml[p], signal: s15.sg[p], hist: s15.hist[p] },
      m5: { line: s5.ml[i], signal: s5.sg[i], hist: s5.hist[i] },
      stBull15: a.st15 === bull,
      stBull5: a.st5 === bull,
      histExpanding: a.expanding,
      score: a.score,
      parts: a.parts,
      conditions: a.checks.map((c) => `${c.label}: ${c.detail}`),
      version: Z15_VERSION,
      outcome,
    })
    lastIdx = i
  }

  // ---- live row at the latest closed 5M candle
  const pl = p15Last >= 0 ? p15Last : c15.length - 1
  const fresh = signals.filter((s) => s.barsAgo < cfg.freshBars).pop() ?? null
  const d15 = dirOf(s15, pl)
  const d5 = dirOf(s5, last)
  const cand: Side | null = fresh ? fresh.side : d15 === 'BULLISH' ? 'BUY' : d15 === 'BEARISH' ? 'SELL' : null
  const crossNow = crossOf(s5, last)
  let checks: Z15Check[] = []
  let score: number | null = null
  let status: Z15Row['status'] = 'WAIT'
  let text = Z15_REASONS.t15
  if (fresh) {
    status = fresh.side
    text = `${fresh.side} — ALL CONDITIONS CONFIRMED, QUALITY ${fresh.score}/100`
    score = fresh.score
    checks = fresh.conditions.map((c, k) => ({ key: 't15' as const, label: c.split(': ')[0], ok: true, detail: `${c.split(': ').slice(1).join(': ')}`, wait: String(k) }))
  } else if (cand) {
    const a = assess(last, pl, cand === 'BUY', crossNow === cand, last - lastIdx)
    checks = a.checks
    const f = a.checks.find((c) => !c.ok)
    if (f) text = f.wait
    else text = Z15_REASONS.m5
    score = f ? null : a.score
  }
  const st15 = s15.stDir[pl] === 1 ? 'BULLISH' : 'BEARISH'
  const st5 = s5.stDir[last] === 1 ? 'BULLISH' : 'BEARISH'
  return {
    signals: [...signals].reverse(),
    rejections: [...rejections].reverse(),
    row: {
      symbol,
      price: c5[last].close,
      barTime: c5[last].openTime + MS5,
      m15: { dir: d15, line: s15.ml[pl], signal: s15.sg[pl], hist: s15.hist[pl] },
      m5: { dir: d5, line: s5.ml[last], signal: s5.sg[last], hist: s5.hist[last] },
      st15,
      st5,
      zero: zeroState(s5, last, s15, pl, cfg),
      agree: d15 !== 'NEUTRAL' && d15 === d5,
      stAgree: st15 === st5,
      status,
      text,
      checks,
      signal: fresh,
      score,
      rejections: [...rejections].reverse().slice(0, 15),
      version: Z15_VERSION,
    },
  }
}

export type Z15Filter = 'all' | 'strongBuy' | 'strongSell' | 'agree' | 'stAgree' | 'score90'

/** Scanner filters. Strong Buy / Strong Sell show only fully qualified signals. */
export function applyZ15Filter(rows: Z15Row[], f: Z15Filter): Z15Row[] {
  switch (f) {
    case 'strongBuy':
      return rows.filter((r) => r.status === 'BUY')
    case 'strongSell':
      return rows.filter((r) => r.status === 'SELL')
    case 'agree':
      return rows.filter((r) => r.agree)
    case 'stAgree':
      return rows.filter((r) => r.stAgree)
    case 'score90':
      return rows.filter((r) => r.score !== null && r.score > 90)
    default:
      return rows
  }
}
