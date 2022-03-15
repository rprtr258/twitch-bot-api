#include <unistd.h>
ssize_t read(int fd, void* buf, size_t count);
ssize_t write(int fd, const void* buf, size_t count);

#include <stdio.h>
int printf(const char *format, ...);

#include <fcntl.h>
int fcntl(int fd, int cmd, ...);

#include <sys/select.h>
int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
//void FD_CLR(int fd, fd_set *set);
//int FD_ISSET(int fd, fd_set *set);
//void FD_SET(int fd, fd_set *set);
//void FD_ZERO(fd_set *set);

#include "lib.c"

i32 main(i32 argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <OAUTH_TOKEN>\n", argv[0]);
        return 1;
    }
    i32 socket_fd = create_socket(argv[1]);
    struct Buffer buffer;
    struct Buffer stdin_buffer;
    skip_welcome_message(socket_fd);
    i32 flags = fcntl(socket_fd, F_GETFL);
    fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK);
    i32 max_fd_plus_one = (socket_fd > STDIN_FILENO ? socket_fd : STDIN_FILENO) + 1;
    for (;;) {
        fd_set read_fds;
        FD_SET(socket_fd, &read_fds);
        FD_SET(STDIN_FILENO, &read_fds);
        exit_if_fail(select(max_fd_plus_one, &read_fds, NULL, NULL, NULL));
        if (FD_ISSET(STDIN_FILENO, &read_fds)) {
            read_buffer(STDIN_FILENO, &stdin_buffer);
            if (stdin_buffer.size == 0) {
                return 0;
            }
            exit_if_fail(send_part(socket_fd, "PRIVMSG #rprtr258 :"));
            // TODO: do we need to send newline one more time?
            exit_if_fail(write(socket_fd, stdin_buffer.data, stdin_buffer.size));
            exit_if_fail(send_newline(socket_fd));
        }
        if (FD_ISSET(socket_fd, &read_fds)) {
            read_buffer(socket_fd, &buffer);
            if (is_ping_message(&buffer)) {
                send_ping_response(socket_fd);
            }
        }
    }
}

