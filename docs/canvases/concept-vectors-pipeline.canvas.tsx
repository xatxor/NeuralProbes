import {
  BarChart,
  Card,
  CardBody,
  CardHeader,
  CollapsibleSection,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Text,
  computeDAGLayout,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

const PIPELINE_NODES = [
  { id: "dataset", label: "feature_stories", sub: "1036 пар историй" },
  { id: "pair", label: "Одна завязка", sub: "два поведения" },
  { id: "concept", label: "Концепт", sub: "истории A" },
  { id: "antagonist", label: "Антагонист", sub: "истории B" },
  { id: "model", label: "Qwen3-8B", sub: "forward pass" },
  { id: "activations", label: "Активации", sub: "residual stream" },
  { id: "pooling", label: "Pooling", sub: "mean с 50-го токена" },
  { id: "means", label: "Средние", sub: "по историям" },
  { id: "diff", label: "diff", sub: "concept − antagonist" },
  { id: "output", label: "Файлы", sub: ".safetensors" },
];

const PIPELINE_EDGES = [
  { from: "dataset", to: "pair" },
  { from: "pair", to: "concept" },
  { from: "pair", to: "antagonist" },
  { from: "concept", to: "model" },
  { from: "antagonist", to: "model" },
  { from: "model", to: "activations" },
  { from: "activations", to: "pooling" },
  { from: "pooling", to: "means" },
  { from: "means", to: "diff" },
  { from: "diff", to: "output" },
];

const STEPS = [
  {
    id: 0,
    title: "1. Пары историй",
    body: "Берём датасет feature_stories: 1036 пар с одинаковой завязкой, но разным поведением героя.",
    highlight: ["dataset", "pair"],
  },
  {
    id: 1,
    title: "2. Два полюса",
    body: "Каждая пара делится на концепт (желаемое поведение) и антагонист (противоположное). Сюжет, язык и жанр совпадают.",
    highlight: ["concept", "antagonist"],
  },
  {
    id: 2,
    title: "3. Прогон через модель",
    body: "Обе группы историй прогоняются через Qwen3-8B. На каждом слое сохраняются внутренние активации (4096-мерные векторы).",
    highlight: ["model", "activations"],
  },
  {
    id: 3,
    title: "4. Усреднение",
    body: "Для каждой истории берётся среднее по токенам с 50-й позиции. Затем усредняем по всем историям каждого полюса.",
    highlight: ["pooling", "means"],
  },
  {
    id: 4,
    title: "5. Разность = вектор",
    body: "diff = mean(концепт) − mean(антагонист). Общий сюжет сокращается, остаётся только направление поведения.",
    highlight: ["diff"],
  },
  {
    id: 5,
    title: "6. Результат",
    body: "1036 векторов × 5 слоёв (11, 14, 18, 22, 25) сохраняются в diff.safetensors и сопутствующие файлы.",
    highlight: ["output"],
  },
];

const LAYER_STATS = [
  { layer: "11", depth: "0.31", norm: 3.3, reliability: 0.991 },
  { layer: "14", depth: "0.39", norm: 4.4, reliability: 0.993 },
  { layer: "18", depth: "0.50", norm: 6.5, reliability: 0.995 },
  { layer: "22", depth: "0.61", norm: 11.0, reliability: 0.995 },
  { layer: "25", depth: "0.69", norm: 19.8, reliability: 0.995 },
];

function PipelineDiagram({
  activeNodes,
}: {
  activeNodes: Set<string>;
}) {
  const theme = useHostTheme();
  const layout = computeDAGLayout({
    nodes: PIPELINE_NODES.map((n) => ({ id: n.id })),
    edges: PIPELINE_EDGES,
    direction: "vertical",
    nodeWidth: 148,
    nodeHeight: 52,
    rankGap: 56,
    nodeGap: 32,
    padding: 20,
  });

  const nodeMeta = Object.fromEntries(
    PIPELINE_NODES.map((n) => [n.id, n]),
  );

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      style={{ display: "block", maxWidth: 520, margin: "0 auto" }}
    >
      {layout.edges.map((edge) => {
        const active =
          activeNodes.has(edge.from) && activeNodes.has(edge.to);
        const midY = (edge.sourceY + edge.targetY) / 2;
        const path = `M ${edge.sourceX} ${edge.sourceY} C ${edge.sourceX} ${midY}, ${edge.targetX} ${midY}, ${edge.targetX} ${edge.targetY}`;
        return (
          <path
            key={`${edge.from}-${edge.to}`}
            d={path}
            fill="none"
            stroke={active ? theme.accent.primary : theme.stroke.secondary}
            strokeWidth={active ? 2 : 1.5}
            opacity={active ? 1 : 0.45}
          />
        );
      })}

      {layout.nodes.map((node) => {
        const meta = nodeMeta[node.id];
        const active = activeNodes.has(node.id);
        const w = 148;
        const h = 52;
        return (
          <g key={node.id}>
            <rect
              x={node.x}
              y={node.y}
              width={w}
              height={h}
              rx={6}
              fill={active ? theme.fill.secondary : theme.bg.elevated}
              stroke={active ? theme.accent.primary : theme.stroke.primary}
              strokeWidth={active ? 2 : 1}
            />
            <text
              x={node.x + w / 2}
              y={node.y + 20}
              textAnchor="middle"
              fill={active ? theme.text.primary : theme.text.secondary}
              fontSize={12}
              fontWeight={590}
            >
              {meta.label}
            </text>
            <text
              x={node.x + w / 2}
              y={node.y + 38}
              textAnchor="middle"
              fill={theme.text.tertiary}
              fontSize={10}
            >
              {meta.sub}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function VectorSpaceDiagram({ step }: { step: number }) {
  const theme = useHostTheme();
  const w = 420;
  const h = 260;
  const cx = 210;
  const cy = 140;

  const corpus = { x: cx, y: cy };
  const concept = { x: cx + 90, y: cy - 70 };
  const antagonist = { x: cx - 85, y: cy + 55 };

  const showDiff = step >= 4;
  const showCentered = step >= 3;
  const showPoints = step >= 1;

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      {/* axes */}
      <line
        x1={30}
        y1={h - 24}
        x2={w - 20}
        y2={h - 24}
        stroke={theme.stroke.tertiary}
        strokeWidth={1}
      />
      <line
        x1={40}
        y1={h - 30}
        x2={40}
        y2={16}
        stroke={theme.stroke.tertiary}
        strokeWidth={1}
      />
      <text x={w / 2} y={h - 6} textAnchor="middle" fill={theme.text.tertiary} fontSize={10}>
        измерение 1 (из 4096)
      </text>
      <text
        x={12}
        y={h / 2}
        textAnchor="middle"
        fill={theme.text.tertiary}
        fontSize={10}
        transform={`rotate(-90 12 ${h / 2})`}
      >
        измерение 2
      </text>

      {/* grid hint */}
      {[0, 1, 2, 3].map((i) => (
        <circle
          key={i}
          cx={cx}
          cy={cy}
          r={28 + i * 24}
          fill="none"
          stroke={theme.stroke.tertiary}
          strokeWidth={0.5}
          opacity={0.35}
        />
      ))}

      {/* corpus center */}
      {showCentered && (
        <>
          <circle cx={corpus.x} cy={corpus.y} r={5} fill={theme.text.tertiary} />
          <text x={corpus.x + 10} y={corpus.y + 4} fill={theme.text.tertiary} fontSize={10}>
            mean(корпус)
          </text>
        </>
      )}

      {/* concept point */}
      {showPoints && (
        <>
          <line
            x1={corpus.x}
            y1={corpus.y}
            x2={concept.x}
            y2={concept.y}
            stroke={theme.accent.primary}
            strokeWidth={showCentered ? 1.5 : 0}
            strokeDasharray={showCentered ? "4 3" : undefined}
            opacity={0.7}
          />
          <circle cx={concept.x} cy={concept.y} r={7} fill={theme.accent.primary} />
          <text x={concept.x + 10} y={concept.y - 6} fill={theme.accent.primary} fontSize={11} fontWeight={590}>
            концепт
          </text>
          <text x={concept.x + 10} y={concept.y + 10} fill={theme.text.tertiary} fontSize={10}>
            «помогает»
          </text>
        </>
      )}

      {/* antagonist point */}
      {showPoints && (
        <>
          <line
            x1={corpus.x}
            y1={corpus.y}
            x2={antagonist.x}
            y2={antagonist.y}
            stroke={theme.text.secondary}
            strokeWidth={showCentered ? 1.5 : 0}
            strokeDasharray={showCentered ? "4 3" : undefined}
            opacity={0.7}
          />
          <circle cx={antagonist.x} cy={antagonist.y} r={7} fill={theme.text.secondary} />
          <text x={antagonist.x - 10} y={antagonist.y + 22} textAnchor="end" fill={theme.text.secondary} fontSize={11} fontWeight={590}>
            антагонист
          </text>
          <text x={antagonist.x - 10} y={antagonist.y + 36} textAnchor="end" fill={theme.text.tertiary} fontSize={10}>
            «нападает»
          </text>
        </>
      )}

      {/* diff arrow */}
      {showDiff && (
        <>
          <defs>
            <marker
              id="arrowhead"
              markerWidth="8"
              markerHeight={8}
              refX={6}
              refY={3}
              orient="auto"
            >
              <path d="M0,0 L0,6 L8,3 z" fill={theme.text.primary} />
            </marker>
          </defs>
          <line
            x1={antagonist.x + 8}
            y1={antagonist.y - 4}
            x2={concept.x - 8}
            y2={concept.y + 4}
            stroke={theme.text.primary}
            strokeWidth={2.5}
            markerEnd="url(#arrowhead)"
          />
          <rect
            x={cx - 52}
            y={cy - 18}
            width={104}
            height={22}
            rx={4}
            fill={theme.bg.elevated}
            stroke={theme.stroke.primary}
          />
          <text
            x={cx}
            y={cy - 3}
            textAnchor="middle"
            fill={theme.text.primary}
            fontSize={11}
            fontWeight={590}
          >
            diff = A − B
          </text>
        </>
      )}

      {/* story pair label */}
      {step <= 1 && (
        <text x={cx} y={22} textAnchor="middle" fill={theme.text.secondary} fontSize={11}>
          одна завязка → два поведения
        </text>
      )}
    </svg>
  );
}

function StoryPairCard() {
  const theme = useHostTheme();
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 12,
      }}
    >
      <div
        style={{
          padding: 12,
          borderRadius: 6,
          background: theme.fill.tertiary,
          border: `1px solid ${theme.stroke.primary}`,
        }}
      >
        <Text weight="medium" style={{ color: theme.accent.primary, marginBottom: 6 }}>
          Концепт (A)
        </Text>
        <Text style={{ fontSize: 12, color: theme.text.secondary, lineHeight: "18px" }}>
          «Герой встретил незнакомца в лесу и <strong>помог</strong> ему найти дорогу.»
        </Text>
      </div>
      <div
        style={{
          padding: 12,
          borderRadius: 6,
          background: theme.fill.tertiary,
          border: `1px solid ${theme.stroke.primary}`,
        }}
      >
        <Text weight="medium" style={{ color: theme.text.secondary, marginBottom: 6 }}>
          Антагонист (B)
        </Text>
        <Text style={{ fontSize: 12, color: theme.text.secondary, lineHeight: "18px" }}>
          «Герой встретил незнакомца в лесу и <strong>напал</strong> на него.»
        </Text>
      </div>
    </div>
  );
}

export default function ConceptVectorsPipeline() {
  const theme = useHostTheme();
  const [step, setStep] = useCanvasState("step", 0);

  const current = STEPS[step];
  const activeNodes = new Set(current.highlight);

  return (
    <Stack gap={24} style={{ padding: "8px 4px 32px", maxWidth: 760 }}>
      <Stack gap={8}>
        <H1>Как получают concept vectors</H1>
        <Text style={{ color: theme.text.secondary }}>
          Qwen3-8B · josephofthebread/Qwen3-8B-concept-vectors · 1036 пар · 5 слоёв
        </Text>
      </Stack>

      <Row gap={8} wrap>
        {STEPS.map((s) => (
          <button
            key={s.id}
            onClick={() => setStep(s.id)}
            style={{
              border: "none",
              cursor: "pointer",
              padding: "6px 10px",
              borderRadius: 4,
              fontSize: 12,
              background: step === s.id ? theme.accent.control : theme.fill.tertiary,
              color: step === s.id ? theme.text.onAccent : theme.text.secondary,
            }}
          >
            {s.id + 1}
          </button>
        ))}
      </Row>

      <Card>
        <CardHeader>{current.title}</CardHeader>
        <CardBody>
          <Stack gap={16}>
            <Text style={{ color: theme.text.secondary }}>{current.body}</Text>
            {step <= 1 && <StoryPairCard />}
          </Stack>
        </CardBody>
      </Card>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        <Card>
          <CardHeader trailing="DAG">Пайплайн</CardHeader>
          <CardBody>
            <PipelineDiagram activeNodes={activeNodes} />
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing="4096-d">Пространство активаций</CardHeader>
          <CardBody>
            <VectorSpaceDiagram step={step} />
          </CardBody>
        </Card>
      </div>

      <CollapsibleSection
        title="Метрики по слоям"
        trailing="Hugging Face"
        defaultOpen={step >= 5}
      >
        <Stack gap={16}>
          <Row gap={12} wrap>
            <Stat label="Пар" value="1036" />
            <Stat label="Слоёв" value="5" />
            <Stat label="Размерность" value="4096" />
            <Stat label="Reliability" value="≥0.99" tone="success" />
          </Row>

          <Stack gap={6}>
            <H3>Median diff norm по слоям</H3>
            <BarChart
              categories={LAYER_STATS.map((l) => `L${l.layer}`)}
              series={[
                {
                  name: "Norm",
                  data: LAYER_STATS.map((l) => l.norm),
                  tone: "info",
                },
              ]}
              height={180}
            />
            <Text style={{ fontSize: 11, color: theme.text.tertiary }}>
              Source: Hugging Face model card · deeper layers → stronger signal
            </Text>
          </Stack>

          <Row gap={8} wrap>
            {LAYER_STATS.map((l, i) => (
              <span key={l.layer}>
                <Pill size="sm">{`L${l.layer} · depth ${l.depth} · r=${l.reliability}`}</Pill>
              </span>
            ))}
          </Row>
        </Stack>
      </CollapsibleSection>

      <Card>
        <CardHeader>Выходные файлы</CardHeader>
        <CardBody>
          <Stack gap={10}>
            <Row gap={8} align="center" wrap>
              <Pill>diff.safetensors</Pill>
              <Text style={{ fontSize: 12, color: theme.text.secondary }}>
                mean(concept) − mean(antagonist) — главный steering-вектор
              </Text>
            </Row>
            <Row gap={8} align="center" wrap>
              <Pill>concept_centered.safetensors</Pill>
              <Text style={{ fontSize: 12, color: theme.text.secondary }}>
                mean(concept) − mean(corpus)
              </Text>
            </Row>
            <Row gap={8} align="center" wrap>
              <Pill>antagonist_centered.safetensors</Pill>
              <Text style={{ fontSize: 12, color: theme.text.secondary }}>
                mean(antagonist) − mean(corpus)
              </Text>
            </Row>
            <Row gap={8} align="center" wrap>
              <Pill>pairs.parquet</Pill>
              <Text style={{ fontSize: 12, color: theme.text.secondary }}>
                метаданные: названия, классы, надёжность
              </Text>
            </Row>
          </Stack>
        </CardBody>
      </Card>

      <Text style={{ fontSize: 11, color: theme.text.tertiary }}>
        Нажимайте кнопки 1–6 сверху, чтобы пройти пайплайн по шагам.
      </Text>
    </Stack>
  );
}
