use std::{
    env::args,
    io::{Result, Read, ErrorKind::NotFound},
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
    Ok(())
}
