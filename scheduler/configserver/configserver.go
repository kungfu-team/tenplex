package configserver

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strconv"

	kfcs "github.com/lsds/KungFu/srcs/go/kungfu/elastic/configserver"
	"github.com/lsds/KungFu/srcs/go/log"
)

func RunBuiltinConfigServer(port int) {
	const endpoint = `/config`
	addr := net.JoinHostPort("", strconv.Itoa(port))
	log.Infof("running builtin config server listening %s%s", addr, endpoint)
	_, cancel := context.WithCancel(context.TODO())
	defer cancel()
	srv := &http.Server{
		Addr:    addr,
		Handler: logRequest(kfcs.New(cancel, nil, endpoint)),
	}
	srv.SetKeepAlivesEnabled(false)
	if err := srv.ListenAndServe(); err != nil {
		log.Errorf("config server stopped: %v", err)
	}
}

func logRequest(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		log.Debugf("%s %s from %s, UA: %s", req.Method, req.URL.Path, req.RemoteAddr, req.UserAgent())
		h.ServeHTTP(w, req)
	})
}

func Stop(port int) error {
	resp, err := http.Get(fmt.Sprintf("http://localhost:%d/stop", port))
	if err != nil {
		return err
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("stop failed, status code: %d", resp.StatusCode)
	}
	return nil
}
