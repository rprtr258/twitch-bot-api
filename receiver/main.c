#include <unistd.h>
ssize_t read(int fd, void* buf, size_t count);
ssize_t write(int fd, const void* buf, size_t count);

#include <sys/socket.h>
int socket(int family, int type, int protocol);
int connect(int sockfd, const struct sockaddr* addr, socklen_t addrlen);

#include <arpa/inet.h>
in_addr_t inet_addr(const char* cp);

#include <errno.h>
//int errno;

#include <stdlib.h>
void exit(int status);

#include <netdb.h>
struct addrinfo;
int getaddrinfo(const char *node, const char *service, const struct addrinfo *hints, struct addrinfo **res);

#include <stdio.h>
int printf(const char *format, ...);

typedef ssize_t isize;
typedef size_t usize;
typedef int i32;
typedef const char* const_str;
typedef const char str_literal[];
typedef int bool;
const bool TRUE = 1;
const bool FALSE = 0;
#define BOOL(x) ((x) ? TRUE : FALSE)


str_literal NEWLINE = "\n";
const usize BUFFER_CAPACITY = 1024;


static bool are_strings_equal(const_str str1, const_str str2) {
    while (*str1 && *str2) {
        if (*str1 != *str2) {
            return FALSE;
        }
        ++str1;
        ++str2;
    }
    return !BOOL(*str1 || *str2);
}

static i32 send_part(i32 socket_fd, const_str message_part) {
    usize message_part_length = 0;
    {
        const_str cur_char = message_part;
        while (*cur_char) {
            ++cur_char;
            ++message_part_length;
        }
    }
    if (write(socket_fd, message_part, message_part_length) == -1) {
        return errno;
    }
    return 0;
}

static i32 send_newline(i32 socket_fd) {
    if (write(socket_fd, NEWLINE, sizeof(NEWLINE)) == -1) {
        return errno;
    }
    return 0;
}

static void exit_if_fail(i32 err) {
    if (err == -1) {
        exit(errno);
    }
}

i32 main(i32 argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <OAUTH_TOKEN>\n", argv[0]);
        return 1;
    }
    i32 socket_fd;
    {
        // TODO: fix socket flapping
        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd == -1) {
            return errno;
        }
        struct addrinfo* servers = NULL;
        exit_if_fail(getaddrinfo("irc.chat.twitch.tv", "6667", NULL, &servers));
        exit_if_fail(connect(socket_fd, servers->ai_addr, servers->ai_addrlen));
    }
    {
        const_str OAUTH = argv[1];
        exit_if_fail(send_part(socket_fd, "PASS ") == -1);
        exit_if_fail(send_part(socket_fd, OAUTH) == -1);
        exit_if_fail(send_newline(socket_fd) == -1);

        exit_if_fail(send_part(socket_fd, "NICK rprtr258") == -1);
        exit_if_fail(send_newline(socket_fd) == -1);

        exit_if_fail(send_part(socket_fd, "JOIN #rprtr258") == -1);
        exit_if_fail(send_newline(socket_fd) == -1);
    }
    char buffer[BUFFER_CAPACITY];
    {
        // skip welcome message
        exit_if_fail(read(socket_fd, buffer, BUFFER_CAPACITY));
        // skip join channel response
        exit_if_fail(read(socket_fd, buffer, BUFFER_CAPACITY));
    }
    for (;;) {
        isize bytes_read = read(socket_fd, buffer, BUFFER_CAPACITY);
        if (bytes_read == -1) {
            str_literal err_msg = "READ ERROR\n";
            write(STDERR_FILENO, err_msg, sizeof(err_msg));
            return errno;
        }
        if (are_strings_equal(buffer, "PING :tmi.twitch.tv\r\n")) {
            exit_if_fail(send_part(socket_fd, "PONG :tmi.twitch.tv"));
            exit_if_fail(send_newline(socket_fd));
        } else {
            // :rprtr258!rprtr258@rprtr258.tmi.twitch.tv PRIVMSG #rprtr258 :MMMM
            //          ^                                        ^         ^
            //          bang                                     hash      second_colon
            const_str bang_position = buffer + 1;
            while (*bang_position != '!') {
                ++bang_position;
            }
            const_str hash_position = bang_position;
            while (*hash_position != '#') {
                ++hash_position;
            }
            const_str second_colon_position = hash_position;
            while (*second_colon_position != ':') {
                ++second_colon_position;
            }
            // print "{user},{channel},{data}"
            write(STDOUT_FILENO, buffer + 1, bang_position - buffer - 1);
            write(STDOUT_FILENO, ",", 1);
            write(STDOUT_FILENO, hash_position + 1, second_colon_position - hash_position - 2);
            write(STDOUT_FILENO, ",", 1);
            // 4 is ':' in the beginning plus "\r\n\0" in the end
            write(STDOUT_FILENO, second_colon_position + 1, bytes_read - (second_colon_position - buffer) - 4);
            write(STDOUT_FILENO, NEWLINE, sizeof(NEWLINE) - 1);
        }
    }
    return 0;
}

