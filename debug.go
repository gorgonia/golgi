// +build debug

package golgi

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync/atomic"
)

const DEBUG = true

func init() {
	log.Printf("DEBUG")
}

var TABCOUNT uint32
var logger = log.New(os.Stderr, "", 0)
var replacement = "\n"

func tabcount() int {
	return int(atomic.LoadUint32(&TABCOUNT))
}

func enterLogScope() {
	atomic.AddUint32(&TABCOUNT, 1)
	tabcount := tabcount()
	logger.SetPrefix(strings.Repeat("\t", tabcount))
	replacement = "\n" + strings.Repeat("\t", tabcount)
}

func leaveLogScope() {
	tabcount := tabcount()
	tabcount--

	if tabcount < 0 {
		atomic.StoreUint32(&TABCOUNT, 0)
		tabcount = 0
	} else {
		atomic.StoreUint32(&TABCOUNT, uint32(tabcount))
	}
	logger.SetPrefix(strings.Repeat("\t", tabcount))
	replacement = "\n" + strings.Repeat("\t", tabcount)
}

func logf(format string, others ...interface{}) {
	if DEBUG {
		// format = strings.Replace(format, "\n", replacement, -1)
		s := fmt.Sprintf(format, others...)
		s = strings.Replace(s, "\n", replacement, -1)
		logger.Println(s)
		// logger.Printf(format, others...)
	}
}
