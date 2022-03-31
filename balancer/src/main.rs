use std::{
    env::args,
    io::{Result, Read, ErrorKind::NotFound, BufReader, stdin, BufRead, Write},
    process::exit,
    fs::File,
};

fn main() -> Result<()> {
    let argv = args().collect::<Vec<String>>();
    if argv.len() != 2 {
        println!("Usage: {} CONFIG_FILE", argv[0]);
        exit(1);
    }
    let config_filename = &argv[1];
    let mut config_file = match File::open(config_filename) {
        Ok(file) => file,
        Err(err) => {
            if err.kind() == NotFound {
                println!("No such file: {}", config_filename);
            } else {
                println!("{}", err.to_string());
            }
            exit(1);
        },
    };
    let mut config_file_content = String::new();
    config_file.read_to_string(&mut config_file_content)?;
    let mut patterns = Vec::new();
    let mut routes = Vec::new();
    for line in config_file_content.lines() {
        // TODO: there might be ' ' in pattern, split somehow smarter
        let mut splitter = line.split(' ');
        let pattern = splitter.next().expect("Pattern expected");
        patterns.push(&pattern[1..pattern.len()-1]);
        let route_filename = splitter.next().expect("Route expected");
        match File::options().write(true).open(route_filename) {
            Ok(file) => routes.push(file),
            Err(err) => {
                if err.kind() == NotFound {
                    println!("No such file: {}", route_filename);
                } else {
                    println!("{}", err.to_string());
                }
                exit(1);
            },
        }
    }
    let patterns_set = regex::RegexSet::new(patterns.as_slice())
        .expect("All routes should be correct regular expressions");
    for line_or_err in BufReader::new(stdin()).lines() {
        println!("{:?}", line_or_err);
        let line = line_or_err?;
        let routes = patterns_set
            .matches(&line)
            .into_iter()
            .map(|i| &routes[i]);
        for mut route in routes {
            route.write_all(line.as_bytes())?;
            route.write_all(b"\n")?;
            route.flush()?;
        }
    }
    Ok(())
}
