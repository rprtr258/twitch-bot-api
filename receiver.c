#include <unistd.h>
ssize_t read(int fd, void* buf, size_t count);
ssize_t write(int fd, const void* buf, size_t count);

#include <stdio.h>
int printf(const char *format, ...);

#include "lib.c"

const_str find_char(const_str start, char c) {
    const_str res = start;
    while (*res != c) {
        ++res;
    }
    return res;
}

i32 main(i32 argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <OAUTH_TOKEN>\n", argv[0]);
        return 1;
    }
    i32 socket_fd = create_socket(argv[1]);
    struct Buffer buffer;
    skip_welcome_message(socket_fd);
    for (;;) {
        read_buffer(socket_fd, &buffer);
        if (is_ping_message(&buffer)) {
            send_ping_response(socket_fd);
        } else {
            // :rprtr258!rprtr258@rprtr258.tmi.twitch.tv PRIVMSG #rprtr258 :MMMM
            //          ^                                        ^         ^
            //          bang                                     hash      second_colon
            const_str bang_position = find_char(buffer.data + 1, '!');
            const_str hash_position = find_char(bang_position, '#');
            const_str second_colon_position = find_char(hash_position, ':');
            // print "{user},{channel},{data}"
            write(STDOUT_FILENO, buffer.data + 1, bang_position - buffer.data - 1);
            write(STDOUT_FILENO, ",", 1);
            write(STDOUT_FILENO, hash_position + 1, second_colon_position - hash_position - 2);
            write(STDOUT_FILENO, ",", 1);
            // 4 is ':' in the beginning plus "\r\n\0" in the end
            write(STDOUT_FILENO, second_colon_position + 1, buffer.size - (second_colon_position - buffer.data) - 4);
            write(STDOUT_FILENO, NEWLINE, sizeof(NEWLINE) - 1);
        }
    }
    return 0;
}

