// commit-suggest — native CLI client for the commit message generator API.
//
// Zero external dependencies. Compile to a single binary:
//
//	go build -o commit-suggest .        (Linux/macOS)
//	go build -o commit-suggest.exe .    (Windows)
//
// Usage:
//
//	commit-suggest                  # use staged git diff
//	commit-suggest --stdin          # pipe: git diff --staged | commit-suggest --stdin
//	commit-suggest --diff patch.txt # use a diff file
//	commit-suggest --init           # write current flags to config file
//	commit-suggest --version        # print version
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	version        = "1.0.0"
	defaultAPIURL  = "http://localhost:8000"
	defaultTimeout = 30 * time.Second
)

// -------------------------------------------------------------------------- //
// Config                                                                       //
// -------------------------------------------------------------------------- //

// Config is persisted as JSON in ~/.commit-suggest.json and can be overridden
// by environment variables or command-line flags.
type Config struct {
	APIURL      string  `json:"api_url"`
	APIKey      string  `json:"api_key"`
	Intent      bool    `json:"intent"`
	Temperature float64 `json:"temperature"`
}

func defaultConfig() Config {
	return Config{
		APIURL:      defaultAPIURL,
		Intent:      true,
		Temperature: 0.8,
	}
}

// configFilePath returns the platform-appropriate path for the config file.
func configFilePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".commit-suggest.json"
	}
	return filepath.Join(home, ".commit-suggest.json")
}

// loadConfig merges defaults → config file → environment variables.
// Command-line flags are applied by main() after this returns.
func loadConfig() Config {
	cfg := defaultConfig()

	data, err := os.ReadFile(configFilePath())
	if err == nil {
		// Silently ignore parse errors so a corrupted file doesn't brick the tool.
		_ = json.Unmarshal(data, &cfg)
	}

	if v := os.Getenv("COMMIT_SUGGEST_API_URL"); v != "" {
		cfg.APIURL = v
	}
	if v := os.Getenv("COMMIT_SUGGEST_API_KEY"); v != "" {
		cfg.APIKey = v
	}

	return cfg
}

// saveConfig writes cfg to the user's config file.
func saveConfig(cfg Config) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	path := configFilePath()
	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	fmt.Printf("Config saved to %s\n", path)
	return nil
}

// -------------------------------------------------------------------------- //
// API client                                                                   //
// -------------------------------------------------------------------------- //

type generateRequest struct {
	Diff        string  `json:"diff"`
	Intent      bool    `json:"intent"`
	Temperature float64 `json:"temperature"`
	MaxLen      int     `json:"max_len"`
	BeamSize    int     `json:"beam_size"`
}

type generateResponse struct {
	Message string `json:"message"`
	Detail  any    `json:"detail,omitempty"` // FastAPI error body
}

// callAPI sends the diff to the inference server and returns the commit message.
func callAPI(cfg Config, diff string, intent bool, temperature float64, beamSize int) (string, error) {
	body, err := json.Marshal(generateRequest{
		Diff:        diff,
		Intent:      intent,
		Temperature: temperature,
		MaxLen:      20,
		BeamSize:    beamSize,
	})
	if err != nil {
		return "", fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, cfg.APIURL+"/generate", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}

	client := &http.Client{Timeout: defaultTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf(
			"could not reach server at %s — is it running?\n  Start it with: uvicorn server.api:app --host 0.0.0.0 --port 8000\n  Error: %w",
			cfg.APIURL, err,
		)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Try to surface the FastAPI detail message.
		var errResp generateResponse
		if json.Unmarshal(respBody, &errResp) == nil && errResp.Detail != nil {
			return "", fmt.Errorf("server error %d: %v", resp.StatusCode, errResp.Detail)
		}
		return "", fmt.Errorf("server error %d: %s", resp.StatusCode, string(respBody))
	}

	var result generateResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}

	return result.Message, nil
}

// -------------------------------------------------------------------------- //
// Git helpers                                                                  //
// -------------------------------------------------------------------------- //

// getStagedDiff runs `git diff --staged` in the current directory.
func getStagedDiff() (string, error) {
	cmd := exec.Command("git", "diff", "--staged")
	out, err := cmd.Output()
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return "", fmt.Errorf("git error: %s", strings.TrimSpace(string(exitErr.Stderr)))
		}
		return "", fmt.Errorf("git not found: %w", err)
	}
	return string(out), nil
}

// -------------------------------------------------------------------------- //
// Entry point                                                                  //
// -------------------------------------------------------------------------- //

func main() {
	cfg := loadConfig()

	// Flags — defaults come from the loaded config so persisted prefs are respected.
	apiURL := flag.String("api", cfg.APIURL, "API server base `URL` (e.g. http://10.0.0.5:8000)")
	apiKey := flag.String("key", cfg.APIKey, "Bearer token for the API server")
	intent := flag.Bool("intent", cfg.Intent, "Prepend conventional commit prefix (fix:/feat:/refactor:/docs:)")
	temp := flag.Float64("temp", cfg.Temperature, "Sampling temperature (0.0 – 2.0)")
	beamSize := flag.Int("beam", 1, "Beam size for decoding (1 = greedy, >1 = beam search)")
	diffFile := flag.String("diff", "", "Read diff from `file` instead of staged changes")
	useStdin := flag.Bool("stdin", false, "Read diff from stdin (pipe: git diff --staged | commit-suggest --stdin)")
	doInit := flag.Bool("init", false, "Save current flags to config file and exit")
	showVer := flag.Bool("version", false, "Print version and exit")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "commit-suggest %s\n\n", version)
		fmt.Fprintf(os.Stderr, "Generates commit messages from your staged git diff.\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n  commit-suggest [flags]\n\nFlags:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nFirst-time setup:\n  commit-suggest --api http://YOUR_SERVER:8000 --init\n")
	}
	flag.Parse()

	if *showVer {
		fmt.Printf("commit-suggest %s\n", version)
		return
	}

	// Apply flag overrides onto config.
	cfg.APIURL = strings.TrimRight(*apiURL, "/")
	cfg.APIKey = *apiKey

	if *doInit {
		cfg.Intent = *intent
		cfg.Temperature = *temp
		if err := saveConfig(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Obtain the diff.
	var diff string
	switch {
	case *useStdin:
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
			os.Exit(1)
		}
		diff = string(data)

	case *diffFile != "":
		data, err := os.ReadFile(*diffFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading %q: %v\n", *diffFile, err)
			os.Exit(1)
		}
		diff = string(data)

	default:
		var err error
		diff, err = getStagedDiff()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}

	diff = strings.TrimSpace(diff)
	if diff == "" {
		fmt.Fprintln(os.Stderr, "Nothing staged. Stage your changes first:\n  git add <files>")
		os.Exit(1)
	}

	message, err := callAPI(cfg, diff, *intent, *temp, *beamSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(message)
}
