import { useEffect, useState } from 'react'
import { getStatus } from '../api'
import { Wifi, WifiOff, RefreshCw } from 'lucide-react'

export default function StatusBanner() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)

  const check = async () => {
    setLoading(true)
    try {
      const data = await getStatus()
      setStatus(data)
    } catch {
      setStatus({ available: false, model: null, message: 'Backend unreachable' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { check() }, [])

  const ok = status?.available

  return (
    <div
      style={{
        background: ok ? 'var(--green-dim)' : 'var(--red-dim)',
        border: `1px solid ${ok ? '#166534' : '#7f1d1d'}`,
        borderRadius: '8px',
        padding: '8px 14px',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        fontSize: '13px',
      }}
    >
      {loading ? (
        <span className="spinner" style={{ width: 14, height: 14 }} />
      ) : ok ? (
        <Wifi size={14} color="var(--green)" />
      ) : (
        <WifiOff size={14} color="var(--red)" />
      )}
      <span style={{ color: ok ? 'var(--green)' : 'var(--red)', fontWeight: 500 }}>
        {loading
          ? 'Checking backend…'
          : ok
          ? `${status.backend} · ${status.model ?? 'connected'}`
          : `Offline — ${status?.message ?? 'Not reachable'}`}
      </span>
      <button
        onClick={check}
        style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex', alignItems: 'center' }}
        title="Refresh status"
      >
        <RefreshCw size={13} />
      </button>
    </div>
  )
}
