#include <stdlib.h>
char* getenv(const char *name);

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
        printf("Usage: %s CHANNEL\n", argv[0]);
        return 1;
    }
    const_str oauth_token = getenv("TWITCH_OAUTH_TOKEN");
    if (oauth_token == NULL) {
        printf("environment variable TWITCH_OAUTH_TOKEN was not provided\n");
        return 1;
    }
    const_str channel = argv[1];
    i32 socket_fd = create_socket(oauth_token, channel);
    skip_welcome_message(socket_fd);
    struct Buffer socket_buffer = create_buffer();
    struct Buffer stdin_buffer = create_buffer();
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
            isize line_end_position;
            // send every line in buffer
            while ((line_end_position = buffer_find_char(&stdin_buffer, 0, '\n')) != -1) {
                exit_if_fail(send_part(socket_fd, "PRIVMSG #"));
                exit_if_fail(send_part(socket_fd, channel));
                exit_if_fail(send_part(socket_fd, " :"));
                write_buffer(socket_fd, &stdin_buffer, 0, line_end_position + 1);
                buffer_pop(&stdin_buffer, line_end_position + 1);
            }
            // TODO: fix echo 'smth' | ./sender
            if (feof(stdin)) {
                return 0;
            }
        }
        if (FD_ISSET(socket_fd, &read_fds)) {
            read_buffer(socket_fd, &socket_buffer);
            isize zero_position;
            // process every line in buffer
            while ((zero_position = buffer_find_char(&socket_buffer, '\0', 0)) != -1) {
                if (is_ping_message(&socket_buffer)) {
                    send_ping_response(socket_fd);
                }
                buffer_pop(&socket_buffer, zero_position + 1);
            }
        }
    }
    return 0;
}

