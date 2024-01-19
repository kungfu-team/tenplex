package uri

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/url"
	"time"
)

var t0 = time.Now()

func checkSAS(filename, sas string) error {
	q, err := url.ParseQuery(sas)
	if err != nil {
		return err
	}
	se, err := parseTime(q.Get(`se`))
	if err != nil {
		return err
	}
	if se.Before(t0) {
		log.Printf("%s expired %s ago", filename, t0.Sub(*se))
	}
	return nil
}

func parseTime(s string) (*time.Time, error) {
	var i struct {
		T time.Time `json:"time"`
	}
	if err := json.Unmarshal([]byte(fmt.Sprintf(`{"time": %q}`, s)), &i); err != nil {
		return nil, err
	}
	return &i.T, nil
}

func Debug(w io.Writer) {
	for sa, sas := range opener.azSAS {
		fmt.Fprintf(w, "%q: %q\n", sa, sas)
	}
}
