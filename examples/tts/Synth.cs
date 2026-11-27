// Batch-synthesizes sentences with the Windows SAPI voices and records the phoneme / word timings next to each wave file.
//   input  (tab separated):  id <TAB> voice name <TAB> text
//   output: <dir>/<id>.wav (16 kHz, 16-bit mono) and <dir>/<id>.ev with one line per event
//     W <start ms> <char position> <char count>             word boundary
//     P <start ms> <duration ms> <phoneme code points...>   phoneme (codes joined by '+'; the SAPI phone set)
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Speech.AudioFormat;
using System.Speech.Synthesis;
using System.Text;
using System.Threading;

public static class Synth
{
    public static int Run(string inputTsv, string dir, int rate, int workers)
    {
        Directory.CreateDirectory(dir);
        var lines = new List<string[]>();
        foreach (var l in File.ReadAllLines(inputTsv, Encoding.UTF8))
        {
            var p = l.Split('\t');
            if (p.Length == 3 && !File.Exists(Path.Combine(dir, p[0] + ".ev"))) lines.Add(p);
        }
        int next = -1, done = 0;
        var threads = new List<Thread>();
        for (int w = 0; w < workers; w++)
        {
            var t = new Thread(() =>
            {
                var voices = new Dictionary<string, SpeechSynthesizer>();
                while (true)
                {
                    int i = Interlocked.Increment(ref next);
                    if (i >= lines.Count) break;
                    var row = lines[i];
                    try
                    {
                        SpeechSynthesizer s;
                        if (!voices.TryGetValue(row[1], out s)) { s = new SpeechSynthesizer(); s.SelectVoice(row[1]); s.Rate = rate; voices[row[1]] = s; }
                        var sb = new StringBuilder();
                        string wav = Path.Combine(dir, row[0] + ".wav");
                        EventHandler<SpeakProgressEventArgs> hp = (o, e) => sb.AppendFormat(CultureInfo.InvariantCulture, "W {0} {1} {2}\n", (int)e.AudioPosition.TotalMilliseconds, e.CharacterPosition, e.CharacterCount);
                        EventHandler<PhonemeReachedEventArgs> hq = (o, e) =>
                        {
                            var codes = new List<string>();
                            foreach (char c in e.Phoneme) codes.Add(((int)c).ToString());
                            sb.AppendFormat(CultureInfo.InvariantCulture, "P {0} {1} {2}\n", (int)e.AudioPosition.TotalMilliseconds, (int)e.Duration.TotalMilliseconds, string.Join("+", codes));
                        };
                        s.SpeakProgress += hp; s.PhonemeReached += hq;
                        s.SetOutputToWaveFile(wav, new SpeechAudioFormatInfo(16000, AudioBitsPerSample.Sixteen, AudioChannel.Mono));
                        s.Speak(row[2]);
                        s.SetOutputToNull();
                        s.SpeakProgress -= hp; s.PhonemeReached -= hq;
                        File.WriteAllText(Path.Combine(dir, row[0] + ".ev"), sb.ToString());
                    }
                    catch (Exception ex) { Console.Error.WriteLine("fail " + row[0] + ": " + ex.Message); }
                    int d = Interlocked.Increment(ref done);
                    if (d % 100 == 0) Console.WriteLine("done " + d + " / " + lines.Count);
                }
            });
            t.SetApartmentState(ApartmentState.MTA);
            t.Start();
            threads.Add(t);
        }
        foreach (var t in threads) t.Join();
        return done;
    }
}
