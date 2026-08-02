use std::process::Command;

use spatialrust::records::StreamingReceipt;

#[test]
fn streams_pcd_through_crop_to_las_with_receipt() {
    let directory =
        std::env::temp_dir().join(format!("spatialrust-cli-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let input = directory.join("input.pcd");
    let output = directory.join("output.las");
    let receipt_path = directory.join("receipt.json");
    std::fs::write(
        &input,
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
         WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n0 0 0\n1 0 0\n2 0 0\n",
    )
    .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_spatialrust-stream"))
        .arg(&input)
        .arg(&output)
        .args(["--chunk-points", "1", "--memory-budget", "4096", "--crop"])
        .args(["0.5", "-1", "-1", "2", "1", "1"])
        .arg("--receipt")
        .arg(&receipt_path)
        .status()
        .unwrap();
    assert!(status.success());

    let cloud = spatialrust::read_las_file(&output).unwrap();
    assert_eq!(cloud.len(), 2);
    let receipt =
        StreamingReceipt::from_json(&std::fs::read_to_string(&receipt_path).unwrap()).unwrap();
    assert_eq!(receipt.input_points(), 3);
    assert_eq!(receipt.output_points(), 2);
    assert_eq!(receipt.chunks_read(), 3);
    assert_eq!(receipt.chunks_written(), 2);

    std::fs::remove_dir_all(directory).unwrap();
}
