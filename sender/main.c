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

#include <stdio.h>
int printf(const char *format, ...);

#include <stdlib.h>
void exit(int status);

#include <netdb.h>
struct addrinfo;
int getaddrinfo(const char *node, const char *service, const struct addrinfo *hints, struct addrinfo **res);

#include <fcntl.h>
int fcntl(int fd, int cmd, ...);

#include <sys/select.h>
int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
//void FD_CLR(int fd, fd_set *set);
//int FD_ISSET(int fd, fd_set *set);
//void FD_SET(int fd, fd_set *set);
//void FD_ZERO(fd_set *set);


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
        if (read(socket_fd, buffer, BUFFER_CAPACITY) == -1) {
            return errno;
        }
        // skip join channel response
        if (read(socket_fd, buffer, BUFFER_CAPACITY) == -1) {
            return errno;
        }
    }
    i32 flags = fcntl(socket_fd, F_GETFL);
    fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK);
    i32 max_fd_plus_one = (socket_fd > STDIN_FILENO ? socket_fd : STDIN_FILENO) + 1;
    for (;;) {
        fd_set read_fds;
        FD_SET(socket_fd, &read_fds);
        FD_SET(STDIN_FILENO, &read_fds);
        exit_if_fail(select(max_fd_plus_one, &read_fds, NULL, NULL, NULL));
        if (FD_ISSET(STDIN_FILENO, &read_fds)) {
            isize bytes_read = read(STDIN_FILENO, buffer, BUFFER_CAPACITY);
            if (bytes_read == -1) {
                return errno;
            } else if (bytes_read == 0) {
                return 0;
            }
            write(STDOUT_FILENO, buffer, bytes_read);
            exit_if_fail(send_part(socket_fd, "PRIVMSG #rprtr258 :"));
            // TODO: do we need to send newline one more time?
            exit_if_fail(write(socket_fd, buffer, bytes_read));
            exit_if_fail(send_newline(socket_fd));
        }
        if (FD_ISSET(socket_fd, &read_fds)) {
            exit_if_fail(read(socket_fd, buffer, BUFFER_CAPACITY));
            if (are_strings_equal(buffer, "PING :tmi.twitch.tv\r\n")) {
                exit_if_fail(send_part(socket_fd, "PONG :tmi.twitch.tv"));
                exit_if_fail(send_newline(socket_fd));
            }
        }
    }
}

