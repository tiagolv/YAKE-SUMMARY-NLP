import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts'

const ACCENT = '#6c8fff'
const GREEN = '#4ade80'

function normaliseKeywords(keywordsObj) {
  if (!keywordsObj) return []
  if (Array.isArray(keywordsObj.keywords)) {
    return keywordsObj.keywords.map(k => [k.keyword, k.score])
  }
  return []
}

function toNumber(val) {
  if (val == null) return null
  if (typeof val === 'number') return val
  if (typeof val === 'object') {
    if (val.coverage != null) return val.coverage
    if (val.score != null) return val.score
    if (val.value != null) return val.value
    const n = Object.values(val).find(v => typeof v === 'number')
    return n ?? null
  }
  return parseFloat(val) || null
}

function StatRow({ label, value, max = 1, color = ACCENT }) {
  const num = toNumber(value)
  if (num == null) return null
  const pct = Math.min((num / max) * 100, 100)
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{label}</span>
        <span className="mono" style={{ fontSize: 12, color }}>{num.toFixed(3)}</span>
      </div>
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

export default function MetricsCard({ metrics, keywords }) {
  if (!metrics) return null

  const kwPairs = normaliseKeywords(keywords)
  const kwData = kwPairs.slice(0, 12).map(([kw, score]) => ({
    name: kw.length > 18 ? kw.slice(0, 16) + '…' : kw,
    score: parseFloat((1 - score).toFixed(3)),
  }))

  // keyword_coverage may be {coverage, found: [...], missing: [...]}
  const coverageRaw = metrics.keyword_coverage
  const coverageNum = toNumber(coverageRaw)
  const coverageFound = typeof coverageRaw === 'object' && Array.isArray(coverageRaw?.found)
    ? coverageRaw.found.length : null
  const coverageTotal = coverageFound != null && typeof coverageRaw === 'object' && Array.isArray(coverageRaw?.missing)
    ? coverageFound + coverageRaw.missing.length : null

  const coverageLabel = coverageFound != null && coverageTotal != null
    ? `Keyword coverage (${coverageFound}/${coverageTotal} found)`
    : 'Keyword coverage'

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="surface-raised" style={{ padding: 16 }}>
        <div className="label" style={{ marginBottom: 12 }}>Summary metrics</div>
        <StatRow label="Compression ratio" value={metrics.compression_ratio} max={1} />
        {coverageNum != null && (
          <StatRow label={coverageLabel} value={coverageNum} max={1} color={GREEN} />
        )}
        {metrics.yake_in_summary != null && (
          <StatRow
            label={`YAKE hits (${metrics.yake_in_summary}/${metrics.yake_total ?? '?'})`}
            value={(metrics.yake_in_summary ?? 0) / Math.max(metrics.yake_total ?? 1, 1)}
            max={1}
          />
        )}
      </div>

      {(metrics.rouge1 != null || metrics.rougeL != null) && (
        <div className="surface-raised" style={{ padding: 16 }}>
          <div className="label" style={{ marginBottom: 12 }}>ROUGE scores</div>
          <StatRow label="ROUGE-1" value={metrics.rouge1} max={1} />
          <StatRow label="ROUGE-2" value={metrics.rouge2} max={1} />
          <StatRow label="ROUGE-L" value={metrics.rougeL} max={1} />
        </div>
      )}

      {kwData.length > 0 && (
        <div className="surface-raised" style={{ padding: 16 }}>
          <div className="label" style={{ marginBottom: 12 }}>Top YAKE keywords (relevance)</div>
          <ResponsiveContainer width="100%" height={kwData.length * 28 + 20}>
            <BarChart layout="vertical" data={kwData} margin={{ top: 0, right: 10, bottom: 0, left: 10 }}>
              <XAxis type="number" domain={[0, 1]} hide />
              <YAxis
                type="category" dataKey="name" width={130}
                tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                axisLine={false} tickLine={false}
              />
              <Tooltip
                contentStyle={{ background: 'var(--bg-raised)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: 'var(--text-primary)' }}
                itemStyle={{ color: ACCENT }}
                formatter={v => [v.toFixed(3), 'relevance']}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {kwData.map((_, i) => <Cell key={i} fill={ACCENT} opacity={1 - i * 0.05} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}