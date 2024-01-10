package utils

import (
	"log"
	"os"
)

func ExitErr(err error) {
	log.Printf("%v", err)
	os.Exit(1)
}
