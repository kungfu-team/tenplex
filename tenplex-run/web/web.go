package web

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

func WithLogReq(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		LogRequest(req)
		h.ServeHTTP(w, req)
	})
}

var LogRequest = func(r *http.Request) {
	accessLog.Printf("%s %s | %s %s", r.Method, r.URL, r.RemoteAddr, r.UserAgent())
}
var accessLog = logger{l: log.New(os.Stderr, "[access] ", 0)}

type logger struct{ l *log.Logger }

func (l *logger) Printf(format string, v ...interface{}) {
	l.l.Output(2, fmt.Sprintf(format, v...))
}
