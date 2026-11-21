// sokoban-solver-01: a push-optimal Sokoban solver.
//
// The levels this competition draws are a hallway with one to four small rooms
// off it, each room holding 3 or 4 boxes and as many goals. A box pushed out of
// a room into the hallway can never come back, so the rooms are independent
// problems: the level's optimum is the sum of the rooms' optima, and the only
// thing joining them is the player's walk between doors.
//
// This program therefore splits each level into rooms, solves every room
// push-optimally on its own, and stitches the rooms' move strings together
// with plain walks. A room is solved by A* over pushes: a state is the boxes'
// squares plus the player's reachable region (named by its lowest square), a
// push costs 1, and the estimate is the cheapest assignment of boxes to goals
// by lone-box push distance, which no solution can beat. Deadlocks are pruned
// before they are searched: squares a lone box can never leave, and frozen
// 2 x 2 blocks of walls and boxes.
//
// Everything is budgeted by work, never by the clock: searches stop after a
// fixed number of expansions, so two runs on one set write the same bytes.
// If a room's optimal search gives up, the room is retried with faster
// weighted searches before the level is tried as a whole: a solution a few
// pushes over the optimum still earns most of the level's points. All work
// is bounded twice over — per instance and per run — so a pathological level
// can cost at most its share and never the set's 60 seconds.
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/**
 * Work budgets, counted in state expansions, never in wall time, so two runs
 * on one set write the same bytes and no run can outgrow its 60 seconds.
 * A room's whole state space is only tens of thousands of states, so the
 * 400,000 cap on the optimal attempt never binds there; the whole-level
 * search is the expensive one (400,000 expansions measured at 24-38s), so
 * its attempts are capped small and the budgets below bound the run.
 */
const ROOM_EXPANSION_CAP = 400_000;
const ROOM_RETRY_EXPANSION_CAP = 100_000;
const WHOLE_LEVEL_EXPANSION_CAP = 60_000;
/** At most this many expansions per instance, so one hard level cannot starve the rest of the set. */
const WORK_BUDGET_PER_INSTANCE = 60_000;
/** At most this many expansions in the whole run, the backstop that keeps the run inside its wall budget. */
const WORK_BUDGET_TOTAL = 150_000;
let workUsedInstance = 0;
let workUsedTotal = 0;

/** The four moves: step, and the letter it writes. */
const DIRS: { dx: number; dy: number; ch: string }[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

/** A level on a board padded with one wall all round, so no index ever runs off it. */
interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const rawH = rows.length;
  const rawW = Math.max(...rows.map((r) => r.length));
  const w = rawW + 2;
  const h = rawH + 2;
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === '#') continue;
      const i = (y + 1) * w + (x + 1);
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * The squares from which a lone box can still be pushed onto some goal.
 * Walked backwards from the goals: a box on p can be pushed onto c (one step
 * in direction d) exactly when p itself is open and the player can stand on
 * p - d, so the live squares are the goals plus every p reached that way.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    for (const { dx, dy } of DIRS) {
      const d = dy * w + dx;
      const p = c - d;
      if (!wall[p] && !wall[p - d] && !live[p]) {
        live[p] = 1;
        queue.push(p);
      }
    }
  }
  return live;
}

interface Room {
  boxes: number[];
  goals: number[];
  firstCell: number;
}

/**
 * Split a level into rooms. The player's row is the hallway, one square wide:
 * every open square of it that is not live is cut away (a box there reaches no
 * goal), and each connected piece of floor left that holds boxes is a room.
 * Null when the level is no set of rooms: a box starts in the hallway, a goal
 * sits outside every piece, or some piece holds a different count of boxes
 * and goals.
 */
function splitRooms(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const firstCell = s;
    const cells = [s];
    part[s] = rooms.length;
    for (let i = 0; i < cells.length; i++) {
      for (const { dx, dy } of DIRS) {
        const q = cells[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          cells.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], firstCell });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  const out = rooms.filter((r) => r.boxes.length > 0);
  for (const r of out) {
    r.boxes.sort((a, b) => a - b);
    r.goals.sort((a, b) => a - b);
  }
  out.sort((a, b) => a.firstCell - b.firstCell);
  return out;
}

/**
 * For each goal, the fewest pushes a lone box needs to reach it from every
 * square, other boxes ignored: the same backwards walk as liveSquares, one
 * goal at a time, counting steps. A box's true pushes to its goal can only be
 * more, so these distances never overestimate.
 */
function pushDistances(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  return goals.map((goal) => {
    const dist = new Int32Array(wall.length).fill(-1);
    const queue = [goal];
    dist[goal] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi];
      for (const { dx, dy } of DIRS) {
        const d = dy * w + dx;
        const p = c - d;
        if (!wall[p] && !wall[p - d] && dist[p] < 0) {
          dist[p] = dist[c] + 1;
          queue.push(p);
        }
      }
    }
    return dist;
  });
}

/** Every bijection of n boxes onto n goals, as lists of goal indexes. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const build = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      build();
      cur.pop();
      used[i] = false;
    }
  };
  build();
  return out;
}

interface SearchResult {
  pushes: number;
  moves: string;
  endPlayer: number;
  expansions: number;
}

/**
 * The solution of a level: A* over pushes, at most expansionCap expansions,
 * ordered by g + weight·h. Weight 1 is push-optimal; a higher weight trades
 * optimality for speed, which is what the retry ladder below uses when the
 * optimal search gives up. A state is the boxes' squares and the player's
 * reachable region (named by its lowest square); a push costs 1. The estimate
 * is the cheapest assignment of boxes to goals by lone-box push distance (or,
 * for many boxes, the sum of each box's cheapest goal, which is looser but
 * just as safe). Deadlocks are pruned as they arise: a box pushed onto a
 * square no lone box can leave, and a box landing in a frozen 2 x 2 block
 * of walls and boxes with a box off its goal. Null when the search gives up
 * or the level has no solution.
 */
function searchPushOptimal(level: Level, expansionCap: number, weight = 1): SearchResult | null {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = level;
  const size = wall.length;
  const n = startBoxes.length;
  // A level whose boxes and goals do not match can never be solved: every
  // box must end on a distinct goal.
  if (goals.length !== n) return null;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  // A numeric key packs the region's lowest square and the boxes' squares in
  // base `size`; it stays exact only while it fits in a double's 53 bits, so
  // a search with many boxes keys its states by string instead.
  const numericKeySafe = (n + 1) * Math.log2(size) < 53;
  const live = liveSquares(level);
  if (startBoxes.some((b) => !live[b])) return null;
  const distToGoal = pushDistances(level);
  const perms = n <= 5 ? permutations(n) : [];
  const estimate = (bs: number[]): number => {
    if (n <= 5) {
      let best = Infinity;
      for (const perm of perms) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = distToGoal[perm[i]][bs[i]];
          if (d < 0) {
            sum = Infinity;
            break;
          }
          sum += d;
        }
        if (sum < best) best = sum;
      }
      return best;
    }
    let sum = 0;
    for (const b of bs) {
      let m = Infinity;
      for (let g = 0; g < n; g++) {
        const d = distToGoal[g][b];
        if (d >= 0 && d < m) m = d;
      }
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };

  // The states in order of discovery: boxes, player square, pushes so far,
  // estimate, parent, and the push that made them from it.
  const stateBoxes: number[][] = [startBoxes.slice()];
  const statePlayer: number[] = [startPlayer];
  const stateG: number[] = [0];
  const fOf = (g: number, h: number): number => g + weight * h;
  const stateF: number[] = [fOf(0, estimate(startBoxes))];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  if (stateF[0] === Infinity) return null;
  const keyOf = (bs: number[], low: number): number | string => {
    if (!numericKeySafe) return low + ':' + bs.join(',');
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };
  const index = new Map<number | string, number>();

  // Flood-fill scratch: two stamp arrays (one for the region a push is made
  // from, one for the region a push lands in) sharing one stamp counter.
  const reachMark = new Int32Array(size);
  const landMark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const regionLow = (from: number, occ: Uint8Array): number => {
    const st = ++stamp;
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    landMark[from] = st;
    while (sp > 0) {
      const p = stack[--sp];
      if (p < low) low = p;
      for (const { dx, dy } of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && !occ[q] && landMark[q] !== st) {
          landMark[q] = st;
          stack[sp++] = q;
        }
      }
    }
    return low;
  };

  // Whether the box just pushed to t sits in a frozen 2 x 2 block: four
  // squares of walls and boxes, a box among them off its goal. No box in
  // such a block can ever be pushed again.
  const occ = new Uint8Array(size);
  const frozenBlock = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
      if ((occ[a] && !goalSet[a]) || (occ[b] && !goalSet[b]) || (occ[c] && !goalSet[c]) || (occ[d] && !goalSet[d])) return true;
    }
    return false;
  };

  // The open list: a heap of [f, g, state], lowest f first, deeper first on
  // ties, then lowest state number, so the order is fixed across runs.
  type Entry = [number, number, number];
  const heap: Entry[] = [];
  const entryLess = (a: Entry, b: Entry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: Entry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!entryLess(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): Entry => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && entryLess(heap[c + 1], heap[c])) c++;
        if (!entryLess(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };

  for (const b of startBoxes) occ[b] = 1;
  const low0 = regionLow(startPlayer, occ);
  for (const b of startBoxes) occ[b] = 0;
  index.set(keyOf(startBoxes, low0), 0);
  heapPush([stateF[0], 0, 0]);

  let expansions = 0;
  let found = -1;
  while (heap.length > 0) {
    const [f, g, s] = heapPop();
    if (f !== stateF[s] || g !== stateG[s]) continue; // an entry a better path replaced
    const boxes = stateBoxes[s];
    if (boxes.every((b) => goalSet[b])) {
      found = s;
      break;
    }
    if (expansions >= expansionCap || workUsedInstance >= WORK_BUDGET_PER_INSTANCE || workUsedTotal >= WORK_BUDGET_TOTAL) return null;
    expansions++;
    workUsedInstance++;
    workUsedTotal++;
    // The squares the player reaches from here, boxes in the way.
    for (const b of boxes) occ[b] = 1;
    const st = ++stamp;
    let sp = 0;
    stack[sp++] = statePlayer[s];
    reachMark[statePlayer[s]] = st;
    while (sp > 0) {
      const p = stack[--sp];
      for (const { dx, dy } of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && !occ[q] && reachMark[q] !== st) {
          reachMark[q] = st;
          stack[sp++] = q;
        }
      }
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (const { dx, dy } of DIRS) {
        const d = dy * w + dx;
        const behind = b - d;
        const dest = b + d;
        if (reachMark[behind] !== st) continue;
        if (wall[dest] || occ[dest] || !live[dest]) continue;
        occ[b] = 0;
        occ[dest] = 1;
        if (frozenBlock(dest)) {
          occ[dest] = 0;
          occ[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((a, b2) => a - b2);
        const low = regionLow(b, occ);
        const key = keyOf(next, low);
        const h = estimate(next);
        occ[dest] = 0;
        occ[b] = 1;
        if (h === Infinity) continue;
        const g2 = g + 1;
        const prev = index.get(key);
        if (prev !== undefined) {
          if (stateG[prev] <= g2) continue;
          stateG[prev] = g2;
          stateF[prev] = fOf(g2, h);
          parent[prev] = s;
          pushFrom[prev] = b;
          pushTo[prev] = dest;
          heapPush([fOf(g2, h), g2, prev]);
        } else {
          const ni = stateBoxes.length;
          index.set(key, ni);
          stateBoxes.push(next);
          statePlayer.push(b);
          stateG.push(g2);
          stateF.push(fOf(g2, h));
          parent.push(s);
          pushFrom.push(b);
          pushTo.push(dest);
          heapPush([fOf(g2, h), g2, ni]);
        }
      }
    }
    for (const b of boxes) occ[b] = 0;
  }
  if (found < 0) return null;

  // The path of pushes, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let curBoxes = startBoxes.slice();
  for (const s of path) {
    const from = pushFrom[s];
    const to = pushTo[s];
    const d = to - from;
    const dir = DIRS.find(({ dx, dy }) => dy * w + dx === d)!;
    for (const b of curBoxes) occ[b] = 1;
    moves += walkTo(level, occ, player, from - d);
    for (const b of curBoxes) occ[b] = 0;
    moves += dir.ch;
    player = from;
    curBoxes = stateBoxes[s];
  }
  return { pushes: path.length, moves, endPlayer: player, expansions };
}

/** The shortest walk from `from` to `to` around walls and boxes, in LURD letters. */
function walkTo(level: Level, occ: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const { dx, dy, ch } of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = ch;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/**
 * A level solved as a whole, with no room split: the fallback when the split
 * fails. The optimal attempt first, then faster weighted attempts: a solved
 * level at a few pushes over the optimum still earns most of its points,
 * while an unsolved one earns none.
 */
function solveWholeLevel(level: Level): string {
  for (const weight of [1, 2, 8]) {
    const solved = searchPushOptimal(level, WHOLE_LEVEL_EXPANSION_CAP, weight);
    if (solved !== null) return solved.moves;
  }
  return '';
}

/**
 * A level solved room by room: each room's push-optimal solution in turn, the
 * other rooms' boxes standing as walls, the player walking on from where the
 * last room left it. When a room's optimal search gives up, the room is tried
 * again with faster weighted searches before the whole level is tried as one
 * problem: a room solved a few pushes over its optimum still earns most of
 * the level's points, while giving up on it earns none.
 */
function solveLevel(text: string): string {
  workUsedInstance = 0;
  const level = parseLevel(text);
  const rooms = splitRooms(level);
  if (rooms === null) return solveWholeLevel(level);
  let moves = '';
  let player = level.player;
  // Each room's boxes where they stand now: unsolved rooms still hold their
  // starting squares, solved rooms hold their goals. The next room's walls
  // are the other rooms' boxes as they stand, not as they started: a walk
  // routed through a square a solved box has left is fine, one routed
  // through a square a box now sits on would push it off its goal.
  const roomBoxes: number[][] = rooms.map((r) => r.boxes.slice());
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const wall = level.wall.slice();
    for (let oi = 0; oi < rooms.length; oi++) {
      if (oi === ri) continue;
      for (const b of roomBoxes[oi]) wall[b] = 1;
    }
    const sub: Level = { w: level.w, h: level.h, wall, goals: room.goals, boxes: room.boxes, player };
    const attempts = [
      { weight: 1, cap: ROOM_EXPANSION_CAP },
      { weight: 2, cap: ROOM_RETRY_EXPANSION_CAP },
      { weight: 8, cap: ROOM_RETRY_EXPANSION_CAP },
    ];
    let solved: SearchResult | null = null;
    for (const a of attempts) {
      solved = searchPushOptimal(sub, a.cap, a.weight);
      if (solved !== null) break;
    }
    if (solved === null) return solveWholeLevel(level);
    moves += solved.moves;
    player = solved.endPlayer;
    roomBoxes[ri] = room.goals.slice();
  }
  return moves;
}

/** Replay moves on the level: true only when every step is legal and every box ends on a goal. */
function replayOk(text: string, moves: string): boolean {
  const level = parseLevel(text);
  const { w, wall } = level;
  const boxSet = new Set(level.boxes);
  const goalSet = new Set(level.goals);
  let player = level.player;
  for (const ch of moves) {
    const dir = DIRS.find((d) => d.ch === ch.toLowerCase());
    if (dir === undefined) return false;
    const d = dir.dy * w + dir.dx;
    const next = player + d;
    if (wall[next]) return false;
    if (boxSet.has(next)) {
      const beyond = next + d;
      if (wall[beyond] || boxSet.has(beyond)) return false;
      boxSet.delete(next);
      boxSet.add(beyond);
    }
    player = next;
  }
  for (const b of boxSet) if (!goalSet.has(b)) return false;
  return true;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    const text = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
    moves = solveLevel(text);
    if (moves !== '' && !replayOk(text, moves)) moves = '';
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
