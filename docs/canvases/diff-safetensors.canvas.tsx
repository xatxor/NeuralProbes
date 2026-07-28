import {
  BarChart,
  Card,
  CardBody,
  CardHeader,
  H1,
  H3,
  Row,
  Select,
  Stack,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

const LAYERS = [11, 14, 18, 22, 25];
const LAYER_NORMS: Record<number, number> = {
  11: 3.3,
  14: 4.4,
  18: 6.5,
  22: 11.0,
  25: 19.8,
};
const L11_NORM = LAYER_NORMS[11];
const RELATIVE_NORMS = Object.fromEntries(
  LAYERS.map((l) => [l, LAYER_NORMS[l] / L11_NORM]),
) as Record<number, number>;

const STEPS = [
  {
    id: 0,
    title: "Исходные данные",
    body: "Датасет feature_stories: 1036 пар. У каждой пары ~1000 историй на полюс (5 языков × 10 жанров × 10 вариантов × 2 модели).",
  },
  {
    id: 1,
    title: "Matched pair",
    body: "concept_text и antagonist_text делят одну завязку — сюжет, язык и жанр совпадают. Расходятся только в поведении героя.",
  },
  {
    id: 2,
    title: "Forward pass",
    body: "Каждая история → chat: user «Write a story.», assistant = текст. Qwen3-8B без system prompt. На слоях 11–25 снимаем residual stream (4096-d на каждый токен).",
  },
  {
    id: 3,
    title: "Pooling по токенам",
    body: "Для одной истории: mean(h[token]) только для token ≥ 50. Так отсекаем промпт и общую завязку, остаётся «поведенческая» часть.",
  },
  {
    id: 4,
    title: "Mean по историям",
    body: "Усредняем pooled-векторы всех concept-историй пары → μ_concept. То же для antagonist → μ_antagonist.",
  },
  {
    id: 5,
    title: "Разность",
    body: "diff[i, L] = μ_concept − μ_antagonist. Общий сюжет сокращается; в векторе остаётся направление «concept vs antagonist».",
  },
  {
    id: 6,
    title: "diff.safetensors",
    body: "1036 таких векторов × 5 слоёв упаковываются в тензор shape (5, 1036, 4096) и сохраняются как diff.safetensors.",
  },
];

const SHARED_SETUP =
  "Герой встретил незнакомца в лесу. Незнакомец выглядел растерянным и ";
const CONCEPT_TAIL = "попросил помощи, и герой терпеливо провёл его к тропе.";
const ANTAGONIST_TAIL = "сделал резкий жест, и герой отвернулся, не ответив.";

function StepNav({
  step,
  onStep,
}: {
  step: number;
  onStep: (n: number) => void;
}) {
  const theme = useHostTheme();
  return (
    <Row gap={6} wrap>
      {STEPS.map((s) => (
        <button
          key={s.id}
          onClick={() => onStep(s.id)}
          style={{
            border: "none",
            cursor: "pointer",
            padding: "5px 9px",
            borderRadius: 4,
            fontSize: 11,
            background: step === s.id ? theme.accent.control : theme.fill.tertiary,
            color: step === s.id ? theme.text.onAccent : theme.text.secondary,
          }}
        >
          {s.id + 1}
        </button>
      ))}
    </Row>
  );
}

function StorySplitDiagram({ step }: { step: number }) {
  const theme = useHostTheme();
  const showSplit = step >= 1;
  const showModel = step >= 2;

  return (
    <svg width="100%" viewBox="0 0 480 200" style={{ display: "block" }}>
      <text x={240} y={16} textAnchor="middle" fill={theme.text.secondary} fontSize={11}>
        pair_id=71 · asking for help vs refusing help
      </text>

      {/* shared prefix */}
      <rect
        x={20}
        y={36}
        width={440}
        height={28}
        rx={4}
        fill={theme.fill.secondary}
        stroke={theme.stroke.primary}
      />
      <text x={30} y={54} fill={theme.text.primary} fontSize={10}>
        {showSplit ? SHARED_SETUP : "…одна и та же завязка для обоих полюсов…"}
      </text>
      {showSplit && (
        <text x={240} y={78} textAnchor="middle" fill={theme.text.tertiary} fontSize={9}>
          общий префикс (сокращается при diff)
        </text>
      )}

      {showSplit && (
        <>
          <rect
            x={20}
            y={88}
            width={210}
            height={36}
            rx={4}
            fill={theme.fill.tertiary}
            stroke={theme.accent.primary}
            strokeWidth={1.5}
          />
          <text x={28} y={104} fill={theme.accent.primary} fontSize={10} fontWeight={590}>
            concept
          </text>
          <text x={28} y={118} fill={theme.text.secondary} fontSize={9}>
            {CONCEPT_TAIL.slice(0, 42)}…
          </text>

          <rect
            x={250}
            y={88}
            width={210}
            height={36}
            rx={4}
            fill={theme.fill.tertiary}
            stroke={theme.stroke.primary}
          />
          <text x={258} y={104} fill={theme.text.secondary} fontSize={10} fontWeight={590}>
            antagonist
          </text>
          <text x={258} y={118} fill={theme.text.secondary} fontSize={9}>
            {ANTAGONIST_TAIL.slice(0, 42)}…
          </text>
        </>
      )}

      {showModel && (
        <>
          <line x1={125} y1={128} x2={125} y2={148} stroke={theme.stroke.secondary} />
          <line x1={355} y1={128} x2={355} y2={148} stroke={theme.stroke.secondary} />
          <rect x={60} y={150} width={130} height={32} rx={4} fill={theme.bg.elevated} stroke={theme.stroke.primary} />
          <text x={125} y={170} textAnchor="middle" fill={theme.text.secondary} fontSize={10}>
            Qwen3-8B → h[t,L]
          </text>
          <rect x={290} y={150} width={130} height={32} rx={4} fill={theme.bg.elevated} stroke={theme.stroke.primary} />
          <text x={355} y={170} textAnchor="middle" fill={theme.text.secondary} fontSize={10}>
            Qwen3-8B → h[t,L]
          </text>
          <text x={240} y={196} textAnchor="middle" fill={theme.text.tertiary} fontSize={9}>
            × ~1000 историй на полюс
          </text>
        </>
      )}
    </svg>
  );
}

function TokenPoolingDiagram({ step }: { step: number }) {
  const theme = useHostTheme();
  const tokenCount = 24;
  const poolFrom = 8; // visual: token 50 ≈ index 8 in strip
  const active = step >= 3;

  return (
    <svg width="100%" viewBox="0 0 480 160" style={{ display: "block" }}>
      <text x={240} y={14} textAnchor="middle" fill={theme.text.secondary} fontSize={11}>
        Pooling одной истории (residual stream, слой L)
      </text>

      {Array.from({ length: tokenCount }).map((_, i) => {
        const pooled = i >= poolFrom;
        const w = 16;
        const x = 24 + i * (w + 2);
        return (
          <g key={i}>
            <rect
              x={x}
              y={28}
              width={w}
              height={28}
              rx={2}
              fill={pooled && active ? theme.accent.primary : theme.fill.tertiary}
              opacity={pooled && active ? 0.85 : pooled ? 0.35 : 0.2}
              stroke={theme.stroke.primary}
            />
            <text
              x={x + w / 2}
              y={46}
              textAnchor="middle"
              fill={pooled && active ? theme.text.onAccent : theme.text.tertiary}
              fontSize={8}
            >
              {i}
            </text>
          </g>
        );
      })}

      <line
        x1={24 + poolFrom * 18 - 1}
        y1={22}
        x2={24 + poolFrom * 18 - 1}
        y2={62}
        stroke={theme.accent.primary}
        strokeWidth={1.5}
        strokeDasharray="3 2"
      />
      <text x={24 + poolFrom * 18 + 4} y={20} fill={theme.accent.primary} fontSize={9}>
        token ≥ 50
      </text>

      <text x={24} y={78} fill={theme.text.tertiary} fontSize={9}>
        серые = промпт + общая завязка (не входят в mean)
      </text>

      {active && (
        <>
          <text x={240} y={98} textAnchor="middle" fill={theme.text.primary} fontSize={11} fontWeight={590}>
            story_vec = mean(h[50], h[51], …, h[T])
          </text>
          <rect x={180} y={108} width={120} height={36} rx={4} fill={theme.fill.secondary} stroke={theme.stroke.primary} />
          <text x={240} y={130} textAnchor="middle" fill={theme.text.secondary} fontSize={10}>
            ∈ ℝ⁴⁰⁹⁶
          </text>
        </>
      )}

      {step >= 4 && (
        <text x={240} y={154} textAnchor="middle" fill={theme.text.tertiary} fontSize={9}>
          затем mean по всем story_vec одного полюса
        </text>
      )}
    </svg>
  );
}

function DiffVectorDiagram({ step, layer }: { step: number; layer: number }) {
  const theme = useHostTheme();
  const w = 480;
  const h = 220;
  const cx = 240;
  const cy = 120;

  const showConceptCloud = step >= 4;
  const showDiff = step >= 5;

  const conceptCentroid = { x: cx + 75, y: cy - 55 };
  const antagonistCentroid = { x: cx - 70, y: cy + 50 };

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      <text x={cx} y={16} textAnchor="middle" fill={theme.text.secondary} fontSize={11}>
        Пространство активаций (2D-срез из 4096) · слой {layer}
      </text>

      <line x1={40} y1={h - 20} x2={w - 20} y2={h - 20} stroke={theme.stroke.tertiary} />
      <line x1={50} y1={h - 14} x2={50} y2={28} stroke={theme.stroke.tertiary} />

      {/* story clouds */}
      {showConceptCloud &&
        [
          [82, -48],
          [95, -62],
          [68, -58],
          [88, -42],
          [72, -50],
        ].map(([dx, dy], i) => (
          <circle
            key={`c${i}`}
            cx={conceptCentroid.x + dx - 75}
            cy={conceptCentroid.y + dy + 55}
            r={4}
            fill={theme.accent.primary}
            opacity={0.35}
          />
        ))}
      {showConceptCloud &&
        [
          [-65, 45],
          [-78, 58],
          [-55, 52],
          [-72, 38],
          [-60, 48],
        ].map(([dx, dy], i) => (
          <circle
            key={`a${i}`}
            cx={antagonistCentroid.x + dx + 70}
            cy={antagonistCentroid.y + dy - 50}
            r={4}
            fill={theme.text.secondary}
            opacity={0.35}
          />
        ))}

      {showConceptCloud && (
        <>
          <circle cx={conceptCentroid.x} cy={conceptCentroid.y} r={8} fill={theme.accent.primary} />
          <text x={conceptCentroid.x + 12} y={conceptCentroid.y - 4} fill={theme.accent.primary} fontSize={10} fontWeight={590}>
            μ_concept
          </text>
          <circle cx={antagonistCentroid.x} cy={antagonistCentroid.y} r={8} fill={theme.text.secondary} />
          <text x={antagonistCentroid.x - 12} y={antagonistCentroid.y + 20} textAnchor="end" fill={theme.text.secondary} fontSize={10} fontWeight={590}>
            μ_antagonist
          </text>
        </>
      )}

      {showDiff && (
        <>
          <defs>
            <marker id="diffArrow" markerWidth="8" markerHeight={8} refX={6} refY={3} orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill={theme.text.primary} />
            </marker>
          </defs>
          <line
            x1={antagonistCentroid.x + 6}
            y1={antagonistCentroid.y - 4}
            x2={conceptCentroid.x - 6}
            y2={conceptCentroid.y + 4}
            stroke={theme.text.primary}
            strokeWidth={2.5}
            markerEnd="url(#diffArrow)"
          />
          <rect x={cx - 90} y={cy - 14} width={180} height={22} rx={4} fill={theme.bg.elevated} stroke={theme.stroke.primary} />
          <text x={cx} y={cy + 2} textAnchor="middle" fill={theme.text.primary} fontSize={11} fontWeight={590}>
            diff = μ_concept − μ_antagonist
          </text>
          <text x={cx} y={h - 4} textAnchor="middle" fill={theme.text.tertiary} fontSize={9}>
            ‖diff‖ ≈ {LAYER_NORMS[layer]} (median, pair median)
          </text>
        </>
      )}
    </svg>
  );
}

export default function DiffSafetensorsCanvas() {
  const theme = useHostTheme();
  const [step, setStep] = useCanvasState("step", 0);
  const [layer, setLayer] = useCanvasState("layer", 18);

  const current = STEPS[step];
  const layerNum = Number(layer);

  return (
    <Stack gap={20} style={{ padding: "8px 4px 32px", maxWidth: 820 }}>
      <Stack gap={6}>
        <H1>Как получился diff.safetensors</H1>
        <Text style={{ color: theme.text.secondary }}>
          AntonKorznikov/feature_stories → Qwen3-8B activations → mean → subtract → safetensors
        </Text>
      </Stack>

      <StepNav step={step} onStep={setStep} />

      <Card>
        <CardHeader trailing={`шаг ${step + 1}/${STEPS.length}`}>{current.title}</CardHeader>
        <CardBody>
          <Text style={{ color: theme.text.secondary }}>{current.body}</Text>
        </CardBody>
      </Card>

      <Row gap={12} align="center" wrap>
        <Text style={{ fontSize: 12, color: theme.text.secondary }}>Слой для визуализации:</Text>
        <Select
          value={String(layer)}
          onChange={(v) => setLayer(Number(v))}
          options={LAYERS.map((l) => ({ value: String(l), label: `Layer ${l}` }))}
        />
      </Row>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card>
          <CardHeader trailing="pair">Истории</CardHeader>
          <CardBody>
            <StorySplitDiagram step={step} />
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing="token ≥ 50">Pooling</CardHeader>
          <CardBody>
            <TokenPoolingDiagram step={step} />
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader trailing="ℝ⁴⁰⁹⁶">Разность векторов</CardHeader>
        <CardBody>
          <DiffVectorDiagram step={step} layer={layerNum} />
        </CardBody>
      </Card>

      {step >= 5 && (
        <Stack gap={8}>
          <H3>Относительный рост median ‖diff‖ (L11 = 1×)</H3>
          <BarChart
            categories={LAYERS.map((l) => `L${l}`)}
            series={[
              {
                name: "‖diff‖ / ‖diff‖@L11",
                data: LAYERS.map((l) => Number(RELATIVE_NORMS[l].toFixed(2))),
                tone: "info",
              },
            ]}
            height={180}
            valueSuffix="×"
            showValues
            referenceLines={[{ value: 1, label: "L11 baseline" }]}
          />
          <Text style={{ fontSize: 11, color: theme.text.tertiary }}>
            Source: Hugging Face model card · median norm по 1036 парам · L25 ≈{" "}
            {RELATIVE_NORMS[25].toFixed(1)}× сильнее L11
          </Text>
        </Stack>
      )}
    </Stack>
  );
}
