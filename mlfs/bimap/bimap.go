package bimap

type BiMap struct {
	f, g map[string]string
}

func New() *BiMap {
	return &BiMap{
		f: make(map[string]string),
		g: make(map[string]string),
	}
}

func (m *BiMap) Add(k, v string) bool {
	// log.Printf("adding %s: %s", k, v)
	_, a := m.f[k]
	_, b := m.g[v]
	if a || b {
		return false
	}
	m.f[k] = v
	m.g[v] = k
	return true
}

func (m *BiMap) Get(k string) (string, bool) {
	v, ok := m.f[k]
	return v, ok
}

func (m *BiMap) RGet(v string) (string, bool) {
	// log.Printf("RGet %s", v)
	k, ok := m.g[v]
	return k, ok
}
