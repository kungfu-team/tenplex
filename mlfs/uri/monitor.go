package uri

import (
	"io"

	"github.com/kungfu-team/mlfs/closer"
	"github.com/kungfu-team/mlfs/iotrace"
)

type monitor struct {
	c    *iotrace.Counter
	name string
}

func newMonitor(name string) *monitor {
	m := &monitor{
		c:    iotrace.NewCounter(),
		name: name,
	}
	go m.Run()
	return m
}

func (m *monitor) Run() {
	iotrace.Monitor(m.c, m.name)
}

func (m *monitor) Trace(r io.ReadCloser) io.ReadCloser {
	r1 := iotrace.TraceReader(r, m.c)
	return closer.ReadClose(r1, r.Close)
}

var (
	httpRangeRate = newMonitor(`http partial download rate: `)
	httpFullRate  = newMonitor(`http full download rate: `)
	fileReadRate  = newMonitor(`file read rate: `)
)

func withHTTPTrace(f io.ReadCloser, bgn, end int64) io.ReadCloser {
	if bgn == 0 && end == -1 {
		f = httpFullRate.Trace(f)
	} else {
		f = httpRangeRate.Trace(f)
	}
	return f
}
