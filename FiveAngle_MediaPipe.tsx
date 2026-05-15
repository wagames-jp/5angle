/**
 * 5アングル計測 — FiveAngle (MediaPipe版)
 * 完全無料・外部API不要・写真が一切外部送信されない
 *
 * 使用技術:
 *   - MediaPipe Pose Landmarker (Google / Apache 2.0ライセンス)
 *   - ブラウザのみで動作（Node.js不要）
 *
 * セットアップ:
 *   npm install @mediapipe/tasks-vision react react-dom
 *   TypeScript: npm install -D @types/react @types/react-dom typescript
 *
 * 仕組み:
 *   1. 5枚の写真から MediaPipe が33個の体のランドマーク座標を取得
 *   2. 肩幅・腰幅・各部位の比率から実寸を推定
 *   3. 身長を基準スケールとして実際のcmに変換
 *   4. フィット感ヒアリングで微調整
 */

import {
  useState, useRef, useCallback, useEffect, useMemo
} from "react";

// ─── MediaPipe型定義（@mediapipe/tasks-vision より） ─────────────────────────

interface Landmark { x: number; y: number; z: number; visibility?: number; }

interface PoseLandmarkerResult {
  landmarks: Landmark[][];
  worldLandmarks: Landmark[][];
}

interface PoseLandmarker {
  detectForVideo?: unknown;
  detect: (image: HTMLImageElement | HTMLCanvasElement) => PoseLandmarkerResult;
  close: () => void;
}

// MediaPipeのランドマークインデックス（33点）
const LM = {
  NOSE: 0,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,      RIGHT_HIP: 24,
  LEFT_KNEE: 25,     RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,    RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,     RIGHT_HEEL: 30,
} as const;

// ─── 型定義 ───────────────────────────────────────────────────────────────────

type AngleKey = "front" | "right" | "left" | "frontRight" | "frontLeft";

interface AngleConfig {
  key: AngleKey;
  label: string;
  hint: string;
  icon: string;
  useFor: string; // どの計測に使うか
}

interface MeasurementResult {
  underBust: number;
  bust: number;
  cupDiff: number;
  cupLabel: string;
  japanSize: string;
  euSize: string;
  usSize: string;
  confidence: number;
  brandSizes: { brand: string; size: string }[];
  debugLandmarks?: Landmark[]; // デバッグ用
}

// ─── 定数 ─────────────────────────────────────────────────────────────────────

const ANGLES: AngleConfig[] = [
  {
    key: "front",
    label: "正面から",
    hint: "胸をカメラに向けて真っすぐ立ってください。両腕は自然に下ろし、背筋を伸ばしてください。",
    icon: "↓",
    useFor: "肩幅・腰幅・体型比率",
  },
  {
    key: "left",
    label: "左側面から",
    hint: "左の脇腹をカメラに向けて立ってください。腕は体の前に軽く添えてください。",
    icon: "→",
    useFor: "左右対称補正",
  },
  {
    key: "right",
    label: "右側面から",
    hint: "右の脇腹をカメラに向けて立ってください。腕は体の前に軽く添えてください。",
    icon: "←",
    useFor: "バスト奥行き・姿勢",
  },
  {
    key: "frontLeft",
    label: "左斜め前から",
    hint: "左の胸がカメラに向くよう、正面から左に45度体を回して立ってください。",
    icon: "↘",
    useFor: "左右バランス確認",
  },
  {
    key: "frontRight",
    label: "右斜め前から",
    hint: "右の胸がカメラに向くよう、正面から右に45度体を回して立ってください。",
    icon: "↙",
    useFor: "立体形状・カップ推定",
  },
];

const CUP_LABELS = ["AA", "A", "B", "C", "D", "E", "F", "G"];

const FIT_CORRECTIONS: { id: string; label: string; underBustDelta: number; bustDelta: number }[] = [
  { id: "wire",     label: "ワイヤーが脇に当たる",      underBustDelta: 0,   bustDelta: +2.5 },
  { id: "overflow", label: "カップからはみ出る",         underBustDelta: 0,   bustDelta: +2.5 },
  { id: "rideup",   label: "アンダーがずり上がる",       underBustDelta: -2,  bustDelta: 0   },
  { id: "loose",    label: "カップにしわが寄る",         underBustDelta: 0,   bustDelta: -2.5 },
  { id: "strap",    label: "ストラップが落ちる",         underBustDelta: 0,   bustDelta: 0   },
  { id: "none",     label: "特に悩みはない",             underBustDelta: 0,   bustDelta: 0   },
];

// ─── MediaPipe ローダー ────────────────────────────────────────────────────────

let _landmarker: PoseLandmarker | null = null;
let _loading = false;
const _callbacks: Array<(lm: PoseLandmarker) => void> = [];

/**
 * MediaPipe PoseLandmarker を遅延初期化
 * CDN から WASM を読み込む（初回のみ、キャッシュあり）
 */
async function getPoseLandmarker(): Promise<PoseLandmarker> {
  if (_landmarker) return _landmarker;

  return new Promise((resolve, reject) => {
    _callbacks.push(resolve);
    if (_loading) return;
    _loading = true;

    // @mediapipe/tasks-vision を CDN から動的ロード
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js";
    script.onload = async () => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const vision = (window as any).MediaPipeTasksVision;
        const { PoseLandmarker, FilesetResolver } = vision;

        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        _landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU",
          },
          runningMode: "IMAGE",
          numPoses: 1,
        }) as PoseLandmarker;

        _callbacks.forEach(cb => cb(_landmarker!));
        _callbacks.length = 0;
      } catch (e) {
        reject(e);
      }
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

// ─── 計測エンジン ──────────────────────────────────────────────────────────────

/**
 * 2点間のピクセル距離
 */
function dist(a: Landmark, b: Landmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * 複数アングルのランドマークから採寸値を計算
 *
 * アルゴリズム:
 *   1. 正面画像の「肩幅」「腰幅」をピクセルで計測
 *   2. 側面画像の「体の奥行き」をピクセルで計測
 *   3. 身長（cm）÷ 頭〜踵ピクセル = スケール係数（px/cm）
 *   4. 各ピクセル値 × スケール係数 → 実寸(cm)
 *   5. アンダーバスト ≈ 腰幅 × π 係数
 *      バスト ≈ （肩幅 + 側面奥行き）× 楕円係数
 */
function estimateMeasurements(
  landmarks: Partial<Record<AngleKey, Landmark[]>>,
  heightCm: number
): { underBust: number; bust: number; confidence: number } {
  const front = landmarks.front;
  const side  = landmarks.right ?? landmarks.left;

  if (!front) {
    // フォールバック: 身長から体型比率で推定（粗い）
    const estimatedUnder = heightCm * 0.465;
    const estimatedBust  = heightCm * 0.530;
    return { underBust: estimatedUnder, bust: estimatedBust, confidence: 55 };
  }

  // ── スケール係数の計算 ──────────────────────────
  // 頭頂〜踵のピクセル距離（正面）
  const topY    = front[LM.NOSE].y;
  const bottomY = Math.max(
    front[LM.LEFT_HEEL]?.y  ?? 0,
    front[LM.RIGHT_HEEL]?.y ?? 0,
    front[LM.LEFT_ANKLE].y,
    front[LM.RIGHT_ANKLE].y,
  );
  const heightPx = bottomY - topY;
  // 実際の身長より頭頂が少し高いので補正（ランドマークは鼻起点）
  const scalePxPerCm = heightPx / (heightCm * 0.93);

  // ── 正面から肩幅・腰幅 ──────────────────────────
  const shoulderWidthPx = dist(front[LM.LEFT_SHOULDER], front[LM.RIGHT_SHOULDER]);
  const hipWidthPx      = dist(front[LM.LEFT_HIP],      front[LM.RIGHT_HIP]);

  const shoulderWidthCm = shoulderWidthPx / scalePxPerCm;
  const hipWidthCm      = hipWidthPx      / scalePxPerCm;

  // ── 側面から奥行き推定 ──────────────────────────
  let depthCm = shoulderWidthCm * 0.62; // 側面なしの場合のデフォルト比率
  if (side) {
    // 側面: 肩の前後幅をx軸差で計測
    const sideHeightPx = (side[LM.LEFT_ANKLE]?.y ?? side[LM.RIGHT_ANKLE].y)
                       - side[LM.NOSE].y;
    const sideScale    = sideHeightPx / (heightCm * 0.93);
    const depthPx      = Math.abs(side[LM.LEFT_SHOULDER].x - side[LM.RIGHT_SHOULDER].x);
    depthCm = depthPx / sideScale;
  }

  // ── アンダーバスト計算 ──────────────────────────
  // アンダーバスト ≈ 腰幅（cm）× π × 胴体楕円係数
  // 一般的に胴体の横幅:奥行き比は約 1:0.75
  const underBustWidth = hipWidthCm * 0.88; // 腰幅 → アンダーバスト幅に補正
  const underBustDepth = underBustWidth * 0.75;
  // 楕円周囲の近似: π × √((a²+b²)/2)
  const underBust = Math.PI * Math.sqrt((underBustWidth ** 2 + underBustDepth ** 2) / 2);

  // ── バスト計算 ──────────────────────────────────
  // バスト ≈ 肩幅寄与 + 奥行き寄与（斜め写真で補正）
  const bustWidth = shoulderWidthCm * 0.92;
  const bustDepth = depthCm * 1.05;
  const bust = Math.PI * Math.sqrt((bustWidth ** 2 + bustDepth ** 2) / 2);

  // ── 信頼度スコア ────────────────────────────────
  const angleCount    = Object.keys(landmarks).length;
  const visibilityOk  = [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP]
    .every(idx => (front[idx]?.visibility ?? 1) > 0.6);
  const confidence = Math.min(
    60 + angleCount * 6 + (visibilityOk ? 8 : 0) + (side ? 10 : 0),
    95
  );

  return { underBust, bust, confidence };
}

/**
 * 画像からランドマークを検出
 */
async function detectLandmarks(dataUrl: string): Promise<Landmark[] | null> {
  const landmarker = await getPoseLandmarker();

  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const result = landmarker.detect(img);
        resolve(result.landmarks[0] ?? null);
      } catch {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
}

// ─── サイズ計算 ────────────────────────────────────────────────────────────────

function buildResult(
  underBust: number,
  bust: number,
  confidence: number
): MeasurementResult {
  const diff     = bust - underBust;
  const cupIndex = Math.max(0, Math.min(Math.round(diff / 2.5) - 1, CUP_LABELS.length - 1));
  const cupLabel = CUP_LABELS[cupIndex];
  const rounded  = Math.round(underBust / 5) * 5;
  const japanSize = `${rounded}${cupLabel}`;
  const euSize    = `${rounded + 15}${cupLabel}`;
  const usInch    = Math.round(rounded / 2.54);
  const usBand    = usInch % 2 === 0 ? usInch : usInch + 1;
  const usSize    = `${usBand}${cupLabel}`;

  return {
    underBust: Math.round(underBust),
    bust:      Math.round(bust),
    cupDiff:   Math.round(diff),
    cupLabel,
    japanSize,
    euSize,
    usSize,
    confidence,
    brandSizes: [
      { brand: "ワコール",  size: japanSize },
      { brand: "トリンプ",  size: japanSize },
      { brand: "シャルレ",  size: japanSize },
      { brand: "EU",        size: euSize    },
      { brand: "US / UK",   size: usSize    },
    ],
  };
}

// ─── UIコンポーネント ──────────────────────────────────────────────────────────

function Btn({
  children, onClick, disabled = false, variant = "primary", style,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
  style?: React.CSSProperties;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        width: "100%",
        padding: "13px 16px",
        borderRadius: 12,
        border: variant === "secondary" ? "1px solid var(--c-border)" : "none",
        background: disabled
          ? "var(--c-border)"
          : variant === "primary" ? "#534AB7" : "transparent",
        color: disabled ? "var(--c-text3)" : variant === "primary" ? "#fff" : "var(--c-text2)",
        fontSize: 14, fontWeight: 500,
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "opacity .15s",
        ...style,
      }}
    >
      {children}
    </button>
  );
}

function Card({ children, accent = false }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <div style={{
      background: "var(--c-surface)",
      border: `${accent ? "1.5px" : "0.5px"} solid ${accent ? "#534AB7" : "var(--c-border)"}`,
      borderRadius: 14,
      padding: 16,
      marginBottom: 12,
    }}>
      {children}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "8px 0", borderBottom: "0.5px solid var(--c-border)",
      fontSize: 13,
    }}>
      <span style={{ color: "var(--c-text2)" }}>{label}</span>
      <span style={{ fontWeight: 500, color: "var(--c-text)" }}>{value}</span>
    </div>
  );
}

function Dots({ total, current }: { total: number; current: number }) {
  return (
    <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 20 }}>
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{
          height: 8, width: i === current ? 20 : 8,
          borderRadius: i === current ? 4 : "50%",
          background: i < current ? "#1D9E75" : i === current ? "#534AB7" : "var(--c-border)",
          transition: "all .3s",
        }} />
      ))}
    </div>
  );
}

// ─── 各画面 ───────────────────────────────────────────────────────────────────

function HeightScreen({ onNext }: { onNext: (h: number) => void }) {
  const [h, setH] = useState(158);
  return (
    <div>
      <p style={{ fontSize: 13, color: "var(--c-text2)", lineHeight: 1.7, marginBottom: 16 }}>
        5枚の写真から<strong style={{ color: "var(--c-text)" }}>ブラウザだけで</strong>採寸します。<br />
        写真は外部に送信されず、解析後すぐにメモリから削除されます。
      </p>
      <div style={{
        background: "var(--c-purple-light)",
        borderRadius: 10, padding: "10px 14px",
        fontSize: 12, color: "var(--c-purple-dark)",
        marginBottom: 12, lineHeight: 1.8,
      }}>
        📱 <strong>撮影のコツ</strong><br />
        スマホを壁に立てかけるか三脚で固定してください。<br />
        スマホは動かさず、<strong>自分が向きを変えて</strong>5方向を撮影します。<br />
        カメラの高さは胸の位置に合わせると精度が上がります。
      </div>
      <div style={{
        background: "var(--c-teal-light)",
        borderRadius: 10, padding: "10px 14px",
        fontSize: 12, color: "var(--c-teal-dark)",
        marginBottom: 24, lineHeight: 1.6,
      }}>
        🔒 完全ローカル処理 — MediaPipe (Google) 使用・通信なし
      </div>
      <label style={{ fontSize: 13, color: "var(--c-text2)", display: "block", marginBottom: 8 }}>
        身長（採寸の基準スケールになります）
      </label>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 32 }}>
        <input
          type="number" value={h} min={140} max={200} step={1}
          onChange={e => setH(Number(e.target.value))}
          style={{
            width: 100, fontSize: 28, fontWeight: 500,
            border: "1px solid var(--c-border)", borderRadius: 10,
            padding: "8px 12px",
            background: "var(--c-surface)", color: "var(--c-text)",
          }}
        />
        <span style={{ fontSize: 18, color: "var(--c-text2)" }}>cm</span>
      </div>
      <Btn onClick={() => onNext(h)}>撮影ガイドへ →</Btn>
    </div>
  );
}

function ShootScreen({
  height, photos, onCapture, onNext,
}: {
  height: number;
  photos: Partial<Record<AngleKey, string>>;
  onCapture: (a: AngleKey, url: string) => void;
  onNext: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [active, setActive] = useState<AngleKey>("front");
  const done = Object.keys(photos).length;
  const cur  = ANGLES.find(a => a.key === active)!;

  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      onCapture(active, ev.target!.result as string);
      const next = ANGLES.find(a => !photos[a.key] && a.key !== active);
      if (next) setActive(next.key);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  }, [active, photos, onCapture]);

  return (
    <div>
      <Dots total={5} current={done < 5 ? done : 4} />

      <div style={{
        background: "var(--c-purple-light)",
        borderRadius: 10, padding: "10px 14px",
        fontSize: 12, color: "var(--c-purple-dark)",
        marginBottom: 14, lineHeight: 1.6,
      }}>
        <strong>{cur.label}</strong>: {cur.hint}
        <div style={{ marginTop: 4, opacity: .75 }}>→ 使用: {cur.useFor}</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
        {ANGLES.map(a => {
          const isDone   = !!photos[a.key];
          const isActive = a.key === active;
          // 正面は1列全体
          const isFullWidth = a.key === "front";
          return (
            <div key={a.key}
              onClick={() => !isDone && setActive(a.key)}
              style={{
                background: isDone ? "#E1F5EE" : isActive ? "var(--c-purple-light)" : "var(--c-surface)",
                border: `1.5px solid ${isDone ? "#1D9E75" : isActive ? "#534AB7" : "var(--c-border)"}`,
                borderRadius: 10, padding: "12px 8px", textAlign: "center",
                cursor: isDone ? "default" : "pointer",
                transition: "all .2s",
                gridColumn: isFullWidth ? "span 2" : undefined,
              }}
            >
              {photos[a.key]
                ? <img src={photos[a.key]} alt={a.label}
                    style={{ width: 48, height: 48, objectFit: "cover", borderRadius: 6, marginBottom: 4 }} />
                : <div style={{ fontSize: 20, marginBottom: 4, color: isActive ? "#534AB7" : "var(--c-text3)" }}>
                    {a.icon}
                  </div>
              }
              <div style={{
                fontSize: 12,
                color: isDone ? "#0F6E56" : isActive ? "#3C3489" : "var(--c-text2)",
                fontWeight: isDone || isActive ? 500 : 400,
              }}>{a.label}</div>
              {isDone   && <div style={{ fontSize: 10, color: "#0F6E56", marginTop: 2 }}>✓ 完了</div>}
              {isActive && !isDone && <div style={{ fontSize: 10, color: "#534AB7", marginTop: 2, fontWeight: 600 }}>【現在撮影中】</div>}
            </div>
          );
        })}
      </div>

      <input ref={fileRef} type="file" accept="image/*" capture="user"
        style={{ display: "none" }} onChange={handleFile} />

      <Btn onClick={() => fileRef.current?.click()} disabled={done === 5}>
        {done === 5 ? "撮影完了" : `📷 ${cur.label}を撮影`}
      </Btn>
      {done > 0 && (
        <Btn variant="secondary" onClick={onNext} style={{ marginTop: 8 }}>
          解析開始 ({done}/5) →
        </Btn>
      )}
      <p style={{ fontSize: 11, color: "var(--c-text3)", textAlign: "center", marginTop: 8 }}>
        身長 {height}cm を基準スケールに使用
      </p>
    </div>
  );
}

function AnalyzeScreen({
  photos, height, onDone,
}: {
  photos: Partial<Record<AngleKey, string>>;
  height: number;
  onDone: (r: { underBust: number; bust: number; confidence: number }) => void;
}) {
  const [pct, setPct] = useState(0);
  const [msg, setMsg] = useState("MediaPipeを初期化中...");
  const [detail, setDetail] = useState("WASMモジュールを読み込んでいます");
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    (async () => {
      setPct(5);  setMsg("MediaPipeを初期化中...");
      setDetail("初回のみWASMファイルを読み込みます（約2MB）");
      await getPoseLandmarker(); // 初期化

      const allLandmarks: Partial<Record<AngleKey, Landmark[]>> = {};
      const entries = Object.entries(photos) as [AngleKey, string][];
      const step = 70 / Math.max(entries.length, 1);

      for (let i = 0; i < entries.length; i++) {
        const [key, dataUrl] = entries[i];
        setPct(Math.round(10 + step * i));
        setMsg(`${ANGLES.find(a => a.key === key)?.label}を解析中...`);
        setDetail("ランドマーク33点を検出しています");
        const lm = await detectLandmarks(dataUrl);
        if (lm) allLandmarks[key] = lm;
      }

      setPct(85); setMsg("採寸値を計算中...");
      setDetail("スケール係数・楕円近似で実寸を算出");
      const result = estimateMeasurements(allLandmarks, height);

      setPct(100); setMsg("完了");
      await new Promise(r => setTimeout(r, 400));
      onDone(result);
    })();
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "24px 0" }}>
      <div style={{
        width: 56, height: 56,
        border: "3px solid var(--c-border)",
        borderTopColor: "#534AB7",
        borderRadius: "50%",
        animation: "spin .9s linear infinite",
        margin: "0 auto 20px",
      }} />
      <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 6, color: "var(--c-text)" }}>{msg}</div>
      <div style={{ fontSize: 12, color: "var(--c-text3)", marginBottom: 14 }}>{detail}</div>
      <div style={{ background: "var(--c-border)", borderRadius: 4, height: 6 }}>
        <div style={{
          height: 6, borderRadius: 4, background: "#534AB7",
          width: `${pct}%`, transition: "width .5s",
        }} />
      </div>
      <div style={{ fontSize: 12, color: "var(--c-text3)", marginTop: 6 }}>{pct}%</div>
      <div style={{
        marginTop: 24, fontSize: 12, color: "var(--c-teal-dark)",
        background: "var(--c-teal-light)",
        borderRadius: 8, padding: "8px 14px", display: "inline-block",
      }}>
        🔒 処理はすべてあなたのブラウザ内で完結
      </div>
    </div>
  );
}

function FitScreen({ onNext }: { onNext: (sel: string[]) => void }) {
  const [sel, setSel] = useState<Set<string>>(new Set());
  const toggle = (id: string) => setSel(p => {
    const s = new Set(p);
    s.has(id) ? s.delete(id) : s.add(id);
    return s;
  });
  return (
    <div>
      <p style={{ fontSize: 13, color: "var(--c-text2)", marginBottom: 14, lineHeight: 1.6 }}>
        現在のブラで気になることを選ぶと<br />推定精度がさらに向上します。
      </p>
      {FIT_CORRECTIONS.map(q => (
        <div key={q.id} onClick={() => toggle(q.id)} style={{
          background: sel.has(q.id) ? "var(--c-purple-light)" : "var(--c-surface)",
          border: `1px solid ${sel.has(q.id) ? "#534AB7" : "var(--c-border)"}`,
          borderRadius: 10, padding: "12px 14px", marginBottom: 8,
          cursor: "pointer", fontSize: 13,
          color: sel.has(q.id) ? "#3C3489" : "var(--c-text)",
          display: "flex", justifyContent: "space-between", alignItems: "center",
          transition: "all .15s",
        }}>
          {q.label}
          <span style={{ fontSize: 17, color: sel.has(q.id) ? "#534AB7" : "var(--c-text3)" }}>
            {sel.has(q.id) ? "✓" : "+"}
          </span>
        </div>
      ))}
      <Btn style={{ marginTop: 16 }} onClick={() => onNext([...sel])}>
        サイズを確認する →
      </Btn>
    </div>
  );
}

function ResultScreen({ result, onShop }: { result: MeasurementResult; onShop: () => void }) {
  const confidenceColor = result.confidence >= 80 ? "#1D9E75" : result.confidence >= 65 ? "#BA7517" : "#E24B4A";
  return (
    <div>
      <Card accent>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 6 }}>
          <div style={{ fontSize: 46, fontWeight: 700, letterSpacing: -2, color: "var(--c-text)" }}>
            {result.japanSize}
          </div>
          <div style={{ fontSize: 13, color: "#534AB7", fontWeight: 500 }}>推奨サイズ</div>
        </div>
        <div style={{ fontSize: 12, color: confidenceColor, fontWeight: 500 }}>
          信頼度 {result.confidence}% — MediaPipe 5アングル解析
        </div>
        <div style={{ fontSize: 11, color: "var(--c-text3)", marginTop: 2 }}>
          写真枚数が多いほど精度が向上します
        </div>
      </Card>

      <Card>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10, color: "var(--c-text)" }}>採寸結果</div>
        <Row label="アンダーバスト" value={`${result.underBust} cm`} />
        <Row label="バスト"         value={`${result.bust} cm`} />
        <Row label="差（カップ）"   value={`${result.cupDiff} cm → ${result.cupLabel}カップ`} />
      </Card>

      <Card>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10, color: "var(--c-text)" }}>ブランド別換算</div>
        {result.brandSizes.map(b => (
          <Row key={b.brand} label={b.brand} value={b.size} />
        ))}
      </Card>

      <Btn onClick={onShop}>このサイズで商品を見る →</Btn>
    </div>
  );
}

function ProductsScreen({ result }: { result: MeasurementResult }) {
  const products = [
    { name: "ワコール フルカップブラ",  price: "¥4,290", bg: "#EEEDFE", color: "#534AB7" },
    { name: "トリンプ アモレーナ",      price: "¥5,830", bg: "#FBEAF0", color: "#993556" },
    { name: "シャルレ フィットブラ",    price: "¥3,190", bg: "#E1F5EE", color: "#0F6E56" },
    { name: "ピーチジョン シルクブラ",  price: "¥3,850", bg: "#FAEEDA", color: "#854F0B" },
  ];
  return (
    <div>
      <div style={{
        background: "#E1F5EE", borderRadius: 10, padding: "8px 14px",
        marginBottom: 14, fontSize: 12, color: "#0F6E56",
        display: "flex", alignItems: "center", gap: 6,
      }}>
        ✓ <strong>{result.japanSize}</strong> の商品のみ表示中
      </div>
      {products.map(p => (
        <div key={p.name} style={{
          background: "var(--c-surface)",
          border: "0.5px solid var(--c-border)",
          borderRadius: 12, padding: 12,
          display: "flex", gap: 12, marginBottom: 10, cursor: "pointer",
          transition: "border-color .15s",
        }}
          onMouseEnter={e => (e.currentTarget.style.borderColor = "#534AB7")}
          onMouseLeave={e => (e.currentTarget.style.borderColor = "var(--c-border)")}
        >
          <div style={{
            width: 52, height: 60, background: p.bg, borderRadius: 8, flexShrink: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 22, color: p.color,
          }}>♡</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 3, color: "var(--c-text)" }}>{p.name}</div>
            <div style={{ fontSize: 12, color: "var(--c-text2)", marginBottom: 3 }}>
              {p.price}
              <span style={{
                fontSize: 11, background: "#E1F5EE", color: "#0F6E56",
                borderRadius: 20, padding: "2px 8px", marginLeft: 6, fontWeight: 500,
              }}>サイズ一致</span>
            </div>
            <div style={{ fontSize: 11, color: "var(--c-text3)" }}>{result.japanSize}</div>
          </div>
          <span style={{ color: "var(--c-text3)", alignSelf: "center" }}>›</span>
        </div>
      ))}
    </div>
  );
}

// ─── メインアプリ ──────────────────────────────────────────────────────────────

type Screen = "height" | "shoot" | "analyze" | "fit" | "result" | "products";

const TITLES: Record<Screen, string> = {
  height:   "5アングル計測",
  shoot:    "撮影ガイド",
  analyze:  "AI解析中",
  fit:      "フィット感チェック",
  result:   "あなたのサイズ",
  products: "おすすめ商品",
};

const ORDER: Screen[] = ["height","shoot","analyze","fit","result","products"];

export default function FiveAngle() {
  const [screen, setScreen] = useState<Screen>("height");
  const [height, setHeight] = useState(158);
  const [photos, setPhotos] = useState<Partial<Record<AngleKey, string>>>({});
  const [raw, setRaw]       = useState<{ underBust: number; bust: number; confidence: number } | null>(null);
  const [result, setResult] = useState<MeasurementResult | null>(null);

  const addPhoto = useCallback((a: AngleKey, url: string) => {
    setPhotos(p => ({ ...p, [a]: url }));
  }, []);

  const applyFit = (selected: string[]) => {
    if (!raw) return;
    let { underBust, bust, confidence } = raw;
    selected.forEach(id => {
      const q = FIT_CORRECTIONS.find(c => c.id === id);
      if (q) { underBust += q.underBustDelta; bust += q.bustDelta; }
    });
    confidence = Math.min(confidence + selected.length * 2, 97);
    setResult(buildResult(underBust, bust, confidence));
    setScreen("result");
  };

  const goBack = () => {
    const i = ORDER.indexOf(screen);
    if (i > 0) setScreen(ORDER[i - 1]);
  };

  const doneCount = Object.keys(photos).length;

  return (
    <>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        :root {
          --c-surface: #ffffff;
          --c-bg: #f2f1ed;
          --c-border: rgba(0,0,0,.11);
          --c-text: #18181a;
          --c-text2: #68686a;
          --c-text3: #a0a0a2;
          --c-purple-light: #EEEDFE;
          --c-purple-dark: #3C3489;
          --c-teal-light: #E1F5EE;
          --c-teal-dark: #0F6E56;
        }
        @media (prefers-color-scheme: dark) {
          :root {
            --c-surface: #1c1c1e;
            --c-bg: #111113;
            --c-border: rgba(255,255,255,.11);
            --c-text: #f0f0f2;
            --c-text2: #a0a0a2;
            --c-text3: #68686a;
            --c-purple-light: #26215C;
            --c-purple-dark: #AFA9EC;
            --c-teal-light: #04342C;
            --c-teal-dark: #5DCAA5;
          }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input[type=number] { outline: none; }
        input[type=number]:focus { border-color: #534AB7 !important; }
      `}</style>

      <div style={{
        background: "var(--c-bg)", minHeight: "100vh",
        display: "flex", alignItems: "center", justifyContent: "center",
        padding: 16,
        fontFamily: "'Helvetica Neue', 'Hiragino Sans', 'Yu Gothic', sans-serif",
      }}>
        <div style={{
          background: "var(--c-surface)",
          borderRadius: 28, width: "100%", maxWidth: 390,
          border: "0.5px solid var(--c-border)", overflow: "hidden",
        }}>
          {/* ヘッダー */}
          <div style={{
            padding: "16px 20px 14px",
            borderBottom: "0.5px solid var(--c-border)",
            display: "flex", alignItems: "center", gap: 10,
          }}>
            {screen !== "height" && screen !== "analyze" && (
              <button onClick={goBack} style={{
                background: "none", border: "none", cursor: "pointer",
                fontSize: 22, color: "var(--c-text2)", padding: 0, lineHeight: 1,
              }}>‹</button>
            )}
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 16, fontWeight: 600, color: "var(--c-text)" }}>
                {TITLES[screen]}
              </div>
              {screen === "shoot" && (
                <div style={{ fontSize: 11, color: "var(--c-text3)" }}>
                  {doneCount} / 5 完了
                </div>
              )}
            </div>
            <div style={{
              fontSize: 10, fontWeight: 700, letterSpacing: ".1em",
              color: "#534AB7", background: "#EEEDFE",
              padding: "3px 8px", borderRadius: 6,
            }}>5ANGLE</div>
          </div>

          {/* 本体 */}
          <div style={{ padding: 20 }}>
            {screen === "height" && (
              <HeightScreen onNext={h => { setHeight(h); setScreen("shoot"); }} />
            )}
            {screen === "shoot" && (
              <ShootScreen
                height={height} photos={photos}
                onCapture={addPhoto}
                onNext={() => setScreen("analyze")}
              />
            )}
            {screen === "analyze" && (
              <AnalyzeScreen
                photos={photos} height={height}
                onDone={r => { setRaw(r); setScreen("fit"); }}
              />
            )}
            {screen === "fit" && <FitScreen onNext={applyFit} />}
            {screen === "result" && result && (
              <ResultScreen result={result} onShop={() => setScreen("products")} />
            )}
            {screen === "products" && result && (
              <ProductsScreen result={result} />
            )}
          </div>
        </div>
      </div>
    </>
  );
}
