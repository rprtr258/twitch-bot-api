#include <stdlib.h>
char* getenv(const char *name);

#include <unistd.h>
ssize_t read(int fd, void* buf, size_t count);
ssize_t write(int fd, const void* buf, size_t count);

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
    usize socket_fd = create_socket(oauth_token, channel);
    skip_welcome_message(socket_fd);
    struct Buffer buffer = create_buffer();
    for (;;) {
        read_buffer(socket_fd, &buffer);
        //read_buffer(STDIN_FILENO, &buffer);
        // processing all messages we read
        for (;;) {
            if (is_ping_message(&buffer)) {
                send_ping_response(socket_fd);
                //send_ping_response(STDOUT_FILENO);
                // remove ping line read from buffer
                buffer_pop(&buffer, sizeof(PING));
            } else {
                // :rprtr258!rprtr258@rprtr258.tmi.twitch.tv PRIVMSG #rprtr258 :MMMM\r\n\0
                //          ^                                        ^         ^         ^
                //          bang                                     hash  second_colon  last
                isize res_or_error;
                usize bang_position;
                res_or_error = buffer_find_char(&buffer, 1, '!');
                if (res_or_error == -1) {
                    break;
                } else {
                    bang_position = (usize)res_or_error;
                }
                usize hash_position;
                res_or_error = buffer_find_char(&buffer, bang_position, '#');
                if (res_or_error == -1) {
                    break;
                } else {
                    hash_position = (usize)res_or_error;
                }
                usize second_colon_position;
                res_or_error = buffer_find_char(&buffer, hash_position, ':');
                if (res_or_error == -1) {
                    break;
                } else {
                    second_colon_position = (usize)res_or_error;
                }
                usize last_position;
                res_or_error = buffer_find_char(&buffer, second_colon_position, '\0');
                if (res_or_error == -1) {
                    break;
                } else {
                    last_position = (usize)res_or_error;
                }
                // print "{user},{channel},{data}"
                write_buffer(STDOUT_FILENO, &buffer, 1, bang_position - 1);
                write(STDOUT_FILENO, ",", 1);
                write_buffer(STDOUT_FILENO, &buffer, hash_position + 1, second_colon_position - hash_position - 2);
                write(STDOUT_FILENO, ",", 1);
                // 3 is ':' in the beginning plus "\r\n" in the end
                write_buffer(STDOUT_FILENO, &buffer, second_colon_position + 1, last_position - second_colon_position - 3);
                send_newline(STDOUT_FILENO);
                // remove line read from buffer
                usize bytes_processed = last_position + 1;
                buffer_pop(&buffer, bytes_processed);
            }
        }
    }
    return 0;
}

