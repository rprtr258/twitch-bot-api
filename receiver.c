#include <unistd.h>
ssize_t read(int fd, void* buf, size_t count);
ssize_t write(int fd, const void* buf, size_t count);

#include <stdio.h>
int printf(const char *format, ...);

#include "lib.c"

usize find_char_in_buffer(struct Buffer *buffer, usize start, char c) {
    usize pos = start;
    while (buffer_get(buffer, pos) != c) {
        ++pos;
    }
    return pos;
}

i32 main(i32 argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <OAUTH_TOKEN>\n", argv[0]);
        return 1;
    }
    i32 socket_fd = create_socket(argv[1]);
    struct Buffer buffer = create_buffer();
    skip_welcome_message(socket_fd);
    for (;;) {
        // TODO: change to reading multiple lines
        read_buffer(socket_fd, &buffer);
        if (is_ping_message(&buffer)) {
            send_ping_response(socket_fd);
            // TODO: remove ping line read from buffer
        } else {
            // TODO: loop until there are messages to process
            // :rprtr258!rprtr258@rprtr258.tmi.twitch.tv PRIVMSG #rprtr258 :MMMM\r\n\0
            //          ^                                        ^         ^         ^
            //          bang                                     hash  second_colon  last
            usize bang_position         = find_char_in_buffer(&buffer,                     1,  '!');
            usize hash_position         = find_char_in_buffer(&buffer,         bang_position,  '#');
            usize second_colon_position = find_char_in_buffer(&buffer,         hash_position,  ':');
            usize last_position         = find_char_in_buffer(&buffer, second_colon_position, '\0');
            // print "{user},{channel},{data}"
            write_buffer(STDOUT_FILENO, &buffer, 1, bang_position - 1);
            write(STDOUT_FILENO, ",", 1);
            write_buffer(STDOUT_FILENO, &buffer, hash_position + 1, second_colon_position - hash_position - 2);
            write(STDOUT_FILENO, ",", 1);
            // 3 is ':' in the beginning plus "\r\n" in the end
            write_buffer(STDOUT_FILENO, &buffer, second_colon_position + 1, last_position - second_colon_position - 3);
            write(STDOUT_FILENO, NEWLINE, sizeof(NEWLINE) - 1);
            // remove line read from buffer
            usize bytes_processed = last_position + 1;
            buffer_pop(&buffer, bytes_processed);
        }
    }
    return 0;
}

