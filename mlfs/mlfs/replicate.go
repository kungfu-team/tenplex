package mlfs

import (
	"bytes"
	"io"
	"net/http"
	"net/url"
)

const (
	X_Replica = `x-replica`
)

func (s *webUI) replicateRequest(w http.ResponseWriter, r *http.Request) error {
	log.Printf("replicate RawQuery: %q", r.URL.RawQuery)
	bs, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}
	log.Printf("body: %d bytes", len(bs))
	for i := 0; i < s.e.redundency+1; i++ {
		id := s.e.peers[(s.e.rank+i)%len(s.e.peers)]
		u := url.URL{
			// Scheme:   r.URL.Scheme,// is empty
			Scheme:   `http`,
			Host:     id.String(),
			Path:     r.URL.Path,
			RawQuery: r.URL.RawQuery,
		}
		req, err := http.NewRequest(r.Method, u.String(), bytes.NewBuffer(bs))
		if err != nil {
			return err
		}
		for k, vs := range r.Header {
			for _, v := range vs {
				req.Header.Add(k, v)
			}
		}
		req.Header.Set(X_Replica, str(s.e.redundency))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		resp.Body.Close()
	}
	return nil
}

func (s *webUI) replicated(f http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		if s.e.redundency > 0 && parseInt(req.Header.Get(X_Replica)) == 0 {
			if err := s.replicateRequest(w, req); err != nil {
				log.Printf("replicateRequest: %v", err)
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		f(w, req)
	}
}
