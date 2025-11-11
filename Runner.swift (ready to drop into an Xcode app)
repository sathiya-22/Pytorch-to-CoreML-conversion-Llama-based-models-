import Foundation
import CoreML

struct LLMManifest: Decodable {
    let layers: Int
    let num_kv_heads: Int
    let head_dim: Int
    let seq_len: Int
    let vocab: Int
    let output_names: [String]
}

final class CoreMLRunner {
    private let model: MLModel
    private let M: LLMManifest

    init(modelName: String) {
        // modelName WITHOUT extension, e.g. "my_llama_legacycache"
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") else {
            fatalError("Missing \(modelName).mlpackage in bundle")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        self.model = try! MLModel(contentsOf: url, configuration: cfg)

        // Load manifest JSON (same base name + _manifest.json)
        guard let manifestURL = Bundle.main.url(forResource: modelName + "_manifest", withExtension: "json") else {
            fatalError("Missing manifest json")
        }
        let data = try! Data(contentsOf: manifestURL)
        self.M = try! JSONDecoder().decode(LLMManifest.self, from: data)
    }

    // Helpers
    private func makeInt32(_ shape: [Int], fill: Int32 = 0) -> MLMultiArray {
        let arr = try! MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        var p = arr.dataPointer.bindMemory(to: Int32.self, capacity: arr.count)
        for _ in 0..<arr.count { p.pointee = fill; p = p.advanced(by: 1) }
        return arr
    }
    private func makeFP16(_ shape: [Int]) -> MLMultiArray {
        // Zero-initialized by default for .float16
        return try! MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float16)
    }

    private func argmax(_ logits: MLMultiArray) -> Int32 {
        let n = logits.count
        var best: Float = -Float.greatestFiniteMagnitude
        var idx: Int32 = 0
        logits.dataPointer.withMemoryRebound(to: UInt16.self, capacity: n) { p in
            for i in 0..<n {
                let f = Float(Float16(bitPattern: p[i]))
                if f > best { best = f; idx = Int32(i) }
            }
        }
        return idx
    }

    private func initialPast() -> [String: MLMultiArray] {
        var d = [String: MLMultiArray]()
        let kshape = [1, M.num_kv_heads, M.seq_len, M.head_dim]
        for i in 0..<M.layers {
            d["past_key_values_\(i)_key"]   = makeFP16(kshape)
            d["past_key_values_\(i)_value"] = makeFP16(kshape)
        }
        return d
    }

    // One forward step: feed last token + KV -> logits + next KV
    private func step(token: Int32, past: inout [String: MLMultiArray]) -> Int32 {
        let inputIds = makeInt32([1,1])
        inputIds[0] = NSNumber(value: token)

        var dict: [String: MLFeatureValue] = ["input_ids": MLFeatureValue(multiArray: inputIds)]
        for (k,v) in past { dict[k] = MLFeatureValue(multiArray: v) }

        let fp = try! MLDictionaryFeatureProvider(dictionary: dict)
        let t0 = CFAbsoluteTimeGetCurrent()
        let out = try! model.prediction(from: fp)
        let ms = Int((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        print("[decode] \(ms) ms")

        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            fatalError("Missing logits")
        }
        let nextId = argmax(logits)

        var newPast = [String: MLMultiArray]()
        for i in 0..<M.layers {
            let kName = "present_key_values_\(i)_key"
            let vName = "present_key_values_\(i)_value"
            guard let kArr = out.featureValue(for: kName)?.multiArrayValue,
                  let vArr = out.featureValue(for: vName)?.multiArrayValue else {
                fatalError("Missing \(kName)/\(vName)")
            }
            newPast["past_key_values_\(i)_key"]   = kArr
            newPast["past_key_values_\(i)_value"] = vArr
        }
        past = newPast
        return nextId
    }

    // Greedy decode
    func generate(ids: [Int32], maxNew: Int) -> [Int32] {
        var past = initialPast()
        var last = ids.first ?? 1
        // Warm cache with prompt
        for t in ids {
            last = step(token: t, past: &past)
        }
        var out = [Int32]()
        while out.count < maxNew {
            let t = step(token: last, past: &past)
            out.append(t)
            last = t
        }
        return out
    }
}
