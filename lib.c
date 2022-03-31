#include <unistd.h>
ssize_t read(int fd, void* buf, size_t count);
ssize_t write(int fd, const void* buf, size_t count);

#include <errno.h>
//int errno;

#include <sys/socket.h>
int socket(int family, int type, int protocol);
int connect(int sockfd, const struct sockaddr* addr, socklen_t addrlen);

#include <arpa/inet.h>
in_addr_t inet_addr(const char* cp);

#include <netdb.h>
struct addrinfo;
int getaddrinfo(const char *node, const char *service, const struct addrinfo *hints, struct addrinfo **res);

#include <stdlib.h>
void exit(int status);

#include <assert.h>
//void assert(scalar);

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
str_literal PING = "PING :tmi.twitch.tv\r\n";

#define exit_if_fail(err) if ((err) == -1) { exit(errno); }

// ========== BUFFER OPS ==========
#define BUFFER_CAPACITY 1024
struct Buffer {
    char data[BUFFER_CAPACITY];
    usize size;
    usize start;
};

static struct Buffer create_buffer(void) {
    struct Buffer buffer;
    buffer.size = 0;
    buffer.start = 0;
    return buffer;
}

static void buffer_pop(struct Buffer* buffer, usize how_many) {
    assert(how_many <= buffer->size);
    buffer->start = (buffer->start + how_many) % BUFFER_CAPACITY;
    buffer->size -= how_many;
}

static char buffer_get(struct Buffer* buffer, usize pos) {
    assert(pos < buffer->size);
    return buffer->data[(buffer->start + pos) % BUFFER_CAPACITY];
}

static void buffer_dump_stdout(struct Buffer* buffer) {
    printf("BUFFER(%lu/%lu): ", buffer->start, buffer->size);
    for (usize i = 0; i < buffer->size; ++i) {
        char c = buffer_get(buffer, i);
        if (c < 0xf) {
            if (c == 0xa) {
                printf("\\n");
            } else if (c == 0xd) {
                printf("\\r");
            } else if (c == 0x0) {
                printf("\\0");
            } else {
                printf("(%x)", c);
            }
        } else {
            printf("%c", c);
        }
    }
    printf("\n");
}

isize buffer_find_char(struct Buffer *buffer, usize start, char c) {
    for (usize pos = start; pos < buffer->size; ++pos) {
        if (buffer_get(buffer, pos) == c) {
            return (isize)pos;
        }
    }
    // not found
    return -1;
}
// ========== BUFFER OPS ==========

static void read_buffer(usize fd, struct Buffer* buffer) {
    assert(buffer->size < BUFFER_CAPACITY);
    usize buffer_end = buffer->start + buffer->size;
    isize read_size;
    if (buffer_end < BUFFER_CAPACITY) {
        usize free_tail_space = BUFFER_CAPACITY - buffer_end;
        read_size = read((int)fd, buffer->data + buffer_end, free_tail_space);
    } else {
        // TODO: read once to temporary buffer, then copy
        usize effective_buffer_end = buffer_end % BUFFER_CAPACITY;
        usize free_space = buffer->start - effective_buffer_end;
        read_size = read((int)fd, buffer->data + effective_buffer_end, free_space);
    }
    exit_if_fail(read_size);
    buffer->size += (usize)read_size;
}

static void write_buffer(usize fd, struct Buffer* buffer, usize from, usize len) {
    usize effective_start = (buffer->start + from) % BUFFER_CAPACITY;
    if (effective_start + len <= BUFFER_CAPACITY) {
        write((int)fd, buffer->data + effective_start, len);
    } else {
        usize tail_size = BUFFER_CAPACITY - effective_start;
        usize head_size = len - tail_size;
        write((int)fd, buffer->data + effective_start, tail_size);
        write((int)fd, buffer->data, head_size);
    }
}

static bool is_ping_message(const struct Buffer* buffer) {
    const_str cur = PING;
    const_str buf_cur = buffer->data + buffer->start;
    const_str buf_end = buffer->data + BUFFER_CAPACITY;
    while (*buf_cur && *cur) {
        if (*buf_cur != *cur) {
            return FALSE;
        }
        buf_cur = (buf_cur == buf_end) ? buffer->data : buf_cur + 1;
        ++cur;
    }
    return !BOOL(*buf_cur || *cur);
}

static void send_part(usize socket_fd, const_str message_part) {
    usize message_part_length = 0;
    {
        const_str cur_char = message_part;
        while (*cur_char) {
            ++cur_char;
            ++message_part_length;
        }
    }
    exit_if_fail(write((int)socket_fd, message_part, message_part_length));
}

static void send_newline(usize socket_fd) {
    exit_if_fail(write((int)socket_fd, NEWLINE, sizeof(NEWLINE)));
}

static usize create_socket(const_str oauth_token, const_str channel) {
    usize socket_fd;
    {
        i32 _socket_fd;
        // TODO: fix socket flapping
        _socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        exit_if_fail(_socket_fd);
        struct addrinfo* servers = NULL;
        exit_if_fail(getaddrinfo("irc.chat.twitch.tv", "6667", NULL, &servers));
        exit_if_fail(connect(_socket_fd, servers->ai_addr, servers->ai_addrlen));
        socket_fd = (usize)_socket_fd;
    }
    {
        send_part(socket_fd, "PASS ");
        send_part(socket_fd, oauth_token);
        send_newline(socket_fd);

        // no matter the nick, we login
        send_part(socket_fd, "NICK tsoding");
        send_newline(socket_fd);

        send_part(socket_fd, "JOIN #");
        send_part(socket_fd, channel);
        send_newline(socket_fd);
    }
    return (usize)socket_fd;
}

void skip_welcome_message(usize socket_fd) {
    char tmp_buffer[BUFFER_CAPACITY];
    // skip welcome message
    exit_if_fail(read((int)socket_fd, tmp_buffer, BUFFER_CAPACITY));
    // skip join channel response
    exit_if_fail(read((int)socket_fd, tmp_buffer, BUFFER_CAPACITY));
}

void send_ping_response(usize socket_fd) {
    send_part(socket_fd, "PONG :tmi.twitch.tv");
    send_newline(socket_fd);
}

