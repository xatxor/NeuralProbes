export type DAGLayoutOptions = {
  nodes: Array<{ id: string }>;
  edges: Array<{ from: string; to: string }>;
  direction?: "vertical" | "horizontal";
  nodeWidth?: number;
  nodeHeight?: number;
  rankGap?: number;
  nodeGap?: number;
  padding?: number;
};

export type DAGLayoutNode = {
  id: string;
  x: number;
  y: number;
  rank: number;
  order: number;
};

export type DAGLayoutEdge = {
  from: string;
  to: string;
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  isBackEdge: boolean;
};

export type DAGLayoutRank = {
  rank: number;
  x: number;
  y: number;
  width: number;
  height: number;
  nodeIds: string[];
};

export type DAGLayoutResult = {
  nodes: DAGLayoutNode[];
  edges: DAGLayoutEdge[];
  ranks: DAGLayoutRank[];
  direction: "vertical" | "horizontal";
  width: number;
  height: number;
};

export function computeDAGLayout(options: DAGLayoutOptions): DAGLayoutResult {
  const {
    nodes,
    edges,
    direction = "vertical",
    nodeWidth = 160,
    nodeHeight = 40,
    rankGap = 64,
    nodeGap = 48,
    padding = 24,
  } = options;

  const ids = nodes.map((n) => n.id);
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const id of ids) {
    incoming.set(id, []);
    outgoing.set(id, []);
  }
  for (const edge of edges) {
    outgoing.get(edge.from)?.push(edge.to);
    incoming.get(edge.to)?.push(edge.from);
  }

  const rank = new Map<string, number>();
  const roots = ids.filter((id) => (incoming.get(id)?.length ?? 0) === 0);
  const queue = [...roots];
  for (const id of ids) rank.set(id, 0);
  for (const id of roots) rank.set(id, 0);

  const seen = new Set<string>();
  while (queue.length) {
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const r = rank.get(id) ?? 0;
    for (const to of outgoing.get(id) ?? []) {
      rank.set(to, Math.max(rank.get(to) ?? 0, r + 1));
      queue.push(to);
    }
  }

  const byRank = new Map<number, string[]>();
  for (const id of ids) {
    const r = rank.get(id) ?? 0;
    if (!byRank.has(r)) byRank.set(r, []);
    byRank.get(r)!.push(id);
  }

  const layoutNodes: DAGLayoutNode[] = [];
  const layoutRanks: DAGLayoutRank[] = [];
  let maxW = 0;
  let maxH = 0;

  const sortedRanks = [...byRank.keys()].sort((a, b) => a - b);
  for (const r of sortedRanks) {
    const nodeIds = byRank.get(r)!;
    const rankWidth = nodeIds.length * nodeWidth + Math.max(0, nodeIds.length - 1) * nodeGap;
    maxW = Math.max(maxW, rankWidth);
  }

  for (const r of sortedRanks) {
    const nodeIds = byRank.get(r)!;
    const rankWidth = nodeIds.length * nodeWidth + Math.max(0, nodeIds.length - 1) * nodeGap;
    const x0 = padding + (maxW - rankWidth) / 2;
    const y0 = padding + r * (nodeHeight + rankGap);

    nodeIds.forEach((id, order) => {
      const x = x0 + order * (nodeWidth + nodeGap);
      layoutNodes.push({ id, x, y: y0, rank: r, order });
    });

    layoutRanks.push({
      rank: r,
      x: x0,
      y: y0,
      width: rankWidth,
      height: nodeHeight,
      nodeIds,
    });
    maxH = Math.max(maxH, y0 + nodeHeight);
  }

  const pos = Object.fromEntries(layoutNodes.map((n) => [n.id, n]));
  const layoutEdges: DAGLayoutEdge[] = edges.map((edge) => {
    const from = pos[edge.from];
    const to = pos[edge.to];
    const vertical = direction === "vertical";
    return {
      ...edge,
      sourceX: from.x + nodeWidth / 2,
      sourceY: vertical ? from.y + nodeHeight : from.y + nodeHeight / 2,
      targetX: to.x + nodeWidth / 2,
      targetY: vertical ? to.y : to.y + nodeHeight / 2,
      isBackEdge: (rank.get(edge.to) ?? 0) <= (rank.get(edge.from) ?? 0),
    };
  });

  return {
    nodes: layoutNodes,
    edges: layoutEdges,
    ranks: layoutRanks,
    direction,
    width: maxW + padding * 2,
    height: maxH + padding,
  };
}
