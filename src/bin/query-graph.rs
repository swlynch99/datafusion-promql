use clap::Parser;
use promql_parser::parser::ast::*;
use promql_parser::parser::parse;

#[derive(Parser)]
#[command(about = "Parse a PromQL query and render its AST as a tree")]
struct Cli {
    /// The PromQL query to parse
    query: String,
}

fn main() {
    let cli = Cli::parse();

    let expr = match parse(&cli.query) {
        Ok(expr) => expr,
        Err(e) => {
            eprintln!("parse error: {e}");
            std::process::exit(1);
        }
    };

    print_expr(&expr, "", true, true);
}

fn print_expr(expr: &Expr, prefix: &str, is_last: bool, is_root: bool) {
    let connector = if is_root {
        ""
    } else if is_last {
        "└── "
    } else {
        "├── "
    };

    println!("{prefix}{connector}{}", node_label(expr));

    let child_prefix = if is_root {
        String::new()
    } else if is_last {
        format!("{prefix}    ")
    } else {
        format!("{prefix}│   ")
    };

    let kids = children(expr);
    for (i, child) in kids.iter().enumerate() {
        let last = i == kids.len() - 1;
        print_expr(child, &child_prefix, last, false);
    }
}

fn node_label(expr: &Expr) -> String {
    match expr {
        Expr::VectorSelector(vs) => format!("{vs}"),
        Expr::MatrixSelector(ms) => format!("{ms}"),
        Expr::Call(call) => format!("{}()", call.func.name),
        Expr::Binary(bin) => format!("{} (binary)", bin.op),
        Expr::Aggregate(agg) => {
            let mut s = agg.op.to_string();
            if let Some(modifier) = &agg.modifier {
                match modifier {
                    LabelModifier::Include(ls) if !ls.is_empty() => {
                        s.push_str(&format!(" by ({ls})"));
                    }
                    LabelModifier::Exclude(ls) => {
                        s.push_str(&format!(" without ({ls})"));
                    }
                    _ => {}
                }
            }
            s
        }
        Expr::Unary(_) => "- (unary)".to_string(),
        Expr::Paren(_) => "(group)".to_string(),
        Expr::Subquery(sq) => {
            let range = sq.range.as_secs();
            match &sq.step {
                Some(step) => format!("subquery [{range}s:{}s]", step.as_secs()),
                None => format!("subquery [{range}s]"),
            }
        }
        Expr::NumberLiteral(n) => n.val.to_string(),
        Expr::StringLiteral(s) => format!("\"{}\"", s.val),
        Expr::Extension(ext) => format!("extension({})", ext.expr.name()),
    }
}

fn children(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::VectorSelector(_)
        | Expr::MatrixSelector(_)
        | Expr::NumberLiteral(_)
        | Expr::StringLiteral(_) => vec![],
        Expr::Call(call) => call.args.args.iter().map(|a| a.as_ref()).collect(),
        Expr::Binary(bin) => vec![&bin.lhs, &bin.rhs],
        Expr::Aggregate(agg) => {
            let mut kids = vec![agg.expr.as_ref()];
            if let Some(param) = &agg.param {
                kids.push(param.as_ref());
            }
            kids
        }
        Expr::Unary(u) => vec![&u.expr],
        Expr::Paren(p) => vec![&p.expr],
        Expr::Subquery(sq) => vec![&sq.expr],
        Expr::Extension(ext) => ext.expr.children().iter().collect(),
    }
}
