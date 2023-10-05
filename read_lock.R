main <- function() {
    pkg_info <- renv:::renv_lockfile_read("renv.lock")$Packages

    pkgs <- c()
    for (i in pkg_info) {
        pkgs <- append(pkgs, i$Package)
    }

    setwd("./renv/cellar")
    pkgInfo <- download.packages(pkgs = pkgs, destdir = getwd(), type = "source")

    setwd("./renv/")
    write.csv(file = "pkgFilenames.csv", basename(pkgInfo[, 2]), row.names = FALSE)
}

main()
