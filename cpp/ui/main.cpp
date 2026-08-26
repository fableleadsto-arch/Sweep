/**
 * Sweep — C++ Terminal UI
 *
 * Interactive OSINT console. Connects to the Python FastAPI backend
 * and uses the C++ engine directly for fast HTML parsing and extraction.
 *
 * Build: make ui
 * Run:   ./sweep_ui [host:port]
 */

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <memory>

// ── HTTP client (minimal, for talking to FastAPI backend) ────────────

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#endif

struct HttpResponse {
    int status = 0;
    std::string body;
    bool ok() const { return status >= 200 && status < 300; }
};

static HttpResponse http_get(const std::string& host, int port, const std::string& path) {
    HttpResponse resp;
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return resp;

    struct hostent* server = gethostbyname(host.c_str());
    if (!server) { close(sock); return resp; }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    addr.sin_port = htons(port);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return resp;
    }

    std::string request = "GET " + path + " HTTP/1.0\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
    send(sock, request.c_str(), request.length(), 0);

    char buffer[4096];
    std::string response;
    int bytes;
    while ((bytes = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes] = '\0';
        response += buffer;
    }
    close(sock);

    // Parse status line
    size_t first_space = response.find(' ');
    if (first_space != std::string::npos) {
        resp.status = std::atoi(response.substr(first_space + 1).c_str());
    }
    size_t body_start = response.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        resp.body = response.substr(body_start + 4);
    }

    return resp;
}

static HttpResponse http_post(const std::string& host, int port, const std::string& path, const std::string& json_body) {
    HttpResponse resp;
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return resp;

    struct hostent* server = gethostbyname(host.c_str());
    if (!server) { close(sock); return resp; }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    addr.sin_port = htons(port);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return resp;
    }

    std::string request =
        "POST " + path + " HTTP/1.0\r\n"
        "Host: " + host + "\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(json_body.length()) + "\r\n"
        "Connection: close\r\n"
        "\r\n" + json_body;

    send(sock, request.c_str(), request.length(), 0);

    char buffer[4096];
    std::string response;
    int bytes;
    while ((bytes = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes] = '\0';
        response += buffer;
    }
    close(sock);

    size_t first_space = response.find(' ');
    if (first_space != std::string::npos) {
        resp.status = std::atoi(response.substr(first_space + 1).c_str());
    }
    size_t body_start = response.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        resp.body = response.substr(body_start + 4);
    }

    return resp;
}

// ── Terminal colors ──────────────────────────────────────────────────

namespace term {
    const std::string RESET   = "\033[0m";
    const std::string BOLD    = "\033[1m";
    const std::string DIM     = "\033[2m";
    const std::string RED     = "\033[31m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN    = "\033[36m";
    const std::string WHITE   = "\033[37m";
    const std::string ORANGE  = "\033[38;5;208m";
}

// ── UI Helpers ───────────────────────────────────────────────────────

static void print_banner() {
    std::cout << term::ORANGE << term::BOLD;
    std::cout << R"(
  ____              _ _
 / ___| _   _  __ _(_) | ___
 \___ \| | | |/ _` | | |/ _ \
  ___) | |_| | (_| | | | (_) |
 |____/ \__, |\__,_|_|_|\___/
        |___/
)" << term::RESET;
    std::cout << term::DIM << "  Python/C++ Web Intelligence Platform" << term::RESET << "\n\n";
}

static void print_help() {
    std::cout << term::CYAN << term::BOLD << "Commands:" << term::RESET << "\n";
    std::cout << "  " << term::YELLOW << "search <query>" << term::RESET << "    — Search the web\n";
    std::cout << "  " << term::YELLOW << "browse <url>" << term::RESET << "     — Browse a page\n";
    std::cout << "  " << term::YELLOW << "extract <url>" << term::RESET << "    — Extract structured data\n";
    std::cout << "  " << term::YELLOW << "research <objective>" << term::RESET << " — Start research run\n";
    std::cout << "  " << term::YELLOW << "status" << term::RESET << "           — Show provider status\n";
    std::cout << "  " << term::YELLOW << "help" << term::RESET << "            — Show this help\n";
    std::cout << "  " << term::YELLOW << "quit" << term::RESET << "            — Exit\n\n";
}

static std::vector<std::string> split_args(const std::string& input) {
    std::vector<std::string> args;
    std::istringstream iss(input);
    std::string arg;
    bool in_quotes = false;
    std::string current;

    for (char c : input) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ' ' && !in_quotes) {
            if (!current.empty()) {
                args.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) args.push_back(current);
    return args;
}

// ── JSON extraction helpers (minimal, no library needed) ─────────────

static std::string json_extract(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    // Skip to value
    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return "";
    pos++;

    // Skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos >= json.length()) return "";

    if (json[pos] == '"') {
        // String value
        pos++;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }

    // Number or boolean
    size_t end = pos;
    while (end < json.length() && json[end] != ',' && json[end] != '}' && json[end] != ']') end++;
    return json.substr(pos, end - pos);
}

// ── Command handlers ─────────────────────────────────────────────────

static void cmd_search(const std::string& host, int port, const std::string& query) {
    std::cout << term::DIM << "Searching: " << query << "..." << term::RESET << "\n";

    // Escape JSON string
    std::string escaped = query;
    std::string json_body = "{\"query\":\"" + escaped + "\",\"limit\":10}";

    auto resp = http_post(host, port, "/api/search", json_body);
    if (!resp.ok()) {
        std::cout << term::RED << "Search failed (HTTP " << resp.status << ")" << term::RESET << "\n";
        return;
    }

    // Parse results (minimal JSON parsing)
    std::string engine = json_extract(resp.body, "engine");
    std::cout << term::GREEN << term::BOLD << "Results via " << engine << term::RESET << "\n\n";

    // Find each result object
    size_t pos = 0;
    int count = 0;
    while ((pos = resp.body.find("\"url\":", pos)) != std::string::npos) {
        std::string url = json_extract(resp.body.substr(pos), "url");
        std::string title = json_extract(resp.body.substr(pos), "title");
        std::string snippet = json_extract(resp.body.substr(pos), "snippet");
        std::string provider = json_extract(resp.body.substr(pos), "provider");

        if (url.empty()) { pos += 6; continue; }

        count++;
        std::cout << term::ORANGE << term::BOLD << count << ". " << title << term::RESET << "\n";
        std::cout << term::DIM << "   " << url << term::RESET << "\n";
        if (!snippet.empty()) {
            std::cout << "   " << snippet.substr(0, 120);
            if (snippet.length() > 120) std::cout << term::DIM << "..." << term::RESET;
            std::cout << "\n";
        }
        std::cout << "   " << term::CYAN << "[" << provider << "]" << term::RESET << "\n\n";

        pos += 6;
        if (count >= 10) break;
    }

    if (count == 0) {
        std::cout << term::DIM << "No results found." << term::RESET << "\n";
    }
}

static void cmd_browse(const std::string& host, int port, const std::string& url) {
    std::cout << term::DIM << "Browsing: " << url << "..." << term::RESET << "\n";

    std::string json_body = "{\"url\":\"" + url + "\",\"max_chars\":8000}";
    auto resp = http_post(host, port, "/api/extract", json_body);
    if (!resp.ok()) {
        std::cout << term::RED << "Browse failed (HTTP " << resp.status << ")" << term::RESET << "\n";
        return;
    }

    std::string title = json_extract(resp.body, "title");
    std::string text = json_extract(resp.body, "text");
    std::string status = json_extract(resp.body, "status");

    std::cout << term::ORANGE << term::BOLD << title << term::RESET << "\n";
    std::cout << term::DIM << "HTTP " << status << " | " << url << term::RESET << "\n\n";

    // Print text, truncated
    if (text.length() > 3000) {
        std::cout << text.substr(0, 3000) << "\n";
        std::cout << term::DIM << "... (truncated, " << text.length() << " chars total)" << term::RESET << "\n";
    } else {
        std::cout << text << "\n";
    }
}

static void cmd_extract(const std::string& host, int port, const std::string& url) {
    std::cout << term::DIM << "Extracting from: " << url << "..." << term::RESET << "\n";

    std::string json_body = "{\"url\":\"" + url + "\"}";
    auto resp = http_post(host, port, "/api/extract", json_body);
    if (!resp.ok()) {
        std::cout << term::RED << "Extract failed (HTTP " << resp.status << ")" << term::RESET << "\n";
        return;
    }

    std::string title = json_extract(resp.body, "title");
    std::string description = json_extract(resp.body, "description");

    std::cout << term::ORANGE << term::BOLD << "Extracted: " << title << term::RESET << "\n";
    if (!description.empty()) {
        std::cout << term::DIM << description << term::RESET << "\n";
    }

    // Show raw JSON for now (TODO: structured display)
    std::cout << "\n" << term::DIM << resp.body.substr(0, 2000) << term::RESET << "\n";
}

static void cmd_research(const std::string& host, int port, const std::string& objective) {
    std::cout << term::MAGENTA << term::BOLD << "Starting research: " << objective << term::RESET << "\n";

    std::string json_body = "{\"objective\":\"" + objective + "\",\"depth\":\"standard\"}";
    auto resp = http_post(host, port, "/api/research", json_body);
    if (!resp.ok()) {
        std::cout << term::RED << "Research failed (HTTP " << resp.status << ")" << term::RESET << "\n";
        return;
    }

    std::string session_id = json_extract(resp.body, "id");
    std::cout << term::GREEN << "Session: " << session_id << term::RESET << "\n";
    std::cout << term::DIM << "Research is running in the background. Polling..." << term::RESET << "\n\n";

    // Poll for results
    for (int i = 0; i < 30; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        auto status_resp = http_get(host, port, "/api/research/" + session_id);
        if (!status_resp.ok()) continue;

        std::string status = json_extract(status_resp.body, "status");
        std::cout << "\r" << term::DIM << "Status: " << status << " (poll " << (i+1) << ")" << term::RESET << std::flush;

        if (status == "complete" || status == "failed") {
            std::cout << "\n";

            // Count evidence
            size_t evidence_pos = status_resp.body.find("\"evidence\":");
            int evidence_count = 0;
            if (evidence_pos != std::string::npos) {
                // Count items in evidence array
                size_t bracket = status_resp.body.find('[', evidence_pos);
                if (bracket != std::string::npos) {
                    for (size_t p = bracket; p < status_resp.body.length(); p++) {
                        if (status_resp.body[p] == '{') evidence_count++;
                    }
                }
            }

            std::cout << term::GREEN << term::BOLD << "Research " << status << "!" << term::RESET << "\n";
            std::cout << "Evidence collected: " << evidence_count << " items\n\n";

            if (status == "complete" && evidence_count > 0) {
                // Show first few evidence items
                size_t ep = 0;
                int shown = 0;
                while ((ep = status_resp.body.find("\"excerpt\":", ep)) != std::string::npos && shown < 5) {
                    std::string excerpt = json_extract(status_resp.body.substr(ep), "excerpt");
                    if (!excerpt.empty()) {
                        std::cout << term::CYAN << "Evidence " << (shown+1) << ":" << term::RESET << "\n";
                        std::cout << "  " << excerpt.substr(0, 200) << "\n\n";
                        shown++;
                    }
                    ep += 10;
                }
            }
            return;
        }
    }
    std::cout << "\n" << term::YELLOW << "Research still running. Check later with: research <id>" << term::RESET << "\n";
}

static void cmd_status(const std::string& host, int port) {
    auto resp = http_get(host, port, "/api/providers");
    if (!resp.ok()) {
        std::cout << term::RED << "Failed to get provider status" << term::RESET << "\n";
        return;
    }

    std::string configured = json_extract(resp.body, "configured");
    std::cout << term::CYAN << term::BOLD << "Search Providers:" << term::RESET << "\n";
    std::cout << "  " << configured << "\n\n";

    auto health = http_get(host, port, "/health");
    if (health.ok()) {
        std::string version = json_extract(health.body, "version");
        std::string python = json_extract(health.body, "python");
        std::cout << term::GREEN << "Backend: " << term::RESET;
        std::cout << "v" << version << " | Python " << python << "\n";
    }
}

// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string host = "127.0.0.1";
    int port = 8787;

    if (argc > 1) {
        std::string arg = argv[1];
        size_t colon = arg.find(':');
        if (colon != std::string::npos) {
            host = arg.substr(0, colon);
            port = std::atoi(arg.substr(colon + 1).c_str());
        } else {
            host = arg;
        }
    }

    print_banner();
    std::cout << term::DIM << "Connected to " << host << ":" << port << term::RESET << "\n";
    print_help();

    std::string line;
    while (true) {
        std::cout << term::ORANGE << "sweep" << term::RESET << term::DIM << " > " << term::RESET << std::flush;
        if (!std::getline(std::cin, line)) break;

        // Trim
        auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line.empty()) continue;

        auto args = split_args(line);
        std::string cmd = args[0];
        std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

        if (cmd == "quit" || cmd == "exit" || cmd == "q") {
            std::cout << term::DIM << "Bye." << term::RESET << "\n";
            break;
        } else if (cmd == "help" || cmd == "h" || cmd == "?") {
            print_help();
        } else if (cmd == "search" || cmd == "s") {
            if (args.size() < 2) {
                std::cout << term::RED << "Usage: search <query>" << term::RESET << "\n";
            } else {
                std::string query;
                for (size_t i = 1; i < args.size(); i++) {
                    if (i > 1) query += " ";
                    query += args[i];
                }
                cmd_search(host, port, query);
            }
        } else if (cmd == "browse" || cmd == "b") {
            if (args.size() < 2) {
                std::cout << term::RED << "Usage: browse <url>" << term::RESET << "\n";
            } else {
                cmd_browse(host, port, args[1]);
            }
        } else if (cmd == "extract" || cmd == "e") {
            if (args.size() < 2) {
                std::cout << term::RED << "Usage: extract <url>" << term::RESET << "\n";
            } else {
                cmd_extract(host, port, args[1]);
            }
        } else if (cmd == "research" || cmd == "r") {
            if (args.size() < 2) {
                std::cout << term::RED << "Usage: research <objective>" << term::RESET << "\n";
            } else {
                std::string objective;
                for (size_t i = 1; i < args.size(); i++) {
                    if (i > 1) objective += " ";
                    objective += args[i];
                }
                cmd_research(host, port, objective);
            }
        } else if (cmd == "status") {
            cmd_status(host, port);
        } else {
            std::cout << term::DIM << "Unknown command: " << cmd << ". Type 'help' for commands." << term::RESET << "\n";
        }
    }

    return 0;
}
