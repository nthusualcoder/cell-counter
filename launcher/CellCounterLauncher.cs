// CellCounter.exe — Cellpose 세포 카운팅 프로그램 런처
// 빌드: launcher\build.bat (Windows 기본 .NET Framework 4 컴파일러 사용, 추가 설치 불필요)
//
// 첫 실행: %LOCALAPPDATA%\CellCounter 에 Python(공식 embeddable 3.10)과 프로그램 코드(GitHub)를 내려받은 뒤
//          app\updater.py --bootstrap 을 실행합니다 (AI 라이브러리·모델 설치 → 프로그램 실행).
// 이후 실행: 이미 설치되어 있으므로 바로 프로그램을 띄웁니다. (업데이트는 프로그램의 [업데이트 확인] 버튼)
// 설치 위치를 바꾸려면 환경변수 CELLCOUNTER_HOME 을 지정하세요.
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Text;

class Launcher
{
    const string Repo = "nthusualcoder/cell-counter";
    const string PythonZipUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip";

    static string Home;

    static int Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.Title = "Cellpose 세포 카운팅 - 런처";
        ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072;  // TLS 1.2

        Home = Environment.GetEnvironmentVariable("CELLCOUNTER_HOME");
        if (string.IsNullOrEmpty(Home))
            Home = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "CellCounter");

        try
        {
            string python = Path.Combine(Home, @"python\python.exe");
            string app = Path.Combine(Home, "app");
            string updater = Path.Combine(app, "updater.py");
            if (!File.Exists(python) || !File.Exists(updater))
            {
                Console.WriteLine("처음 실행합니다. 프로그램을 설치합니다 (인터넷 필요, 10~20분 소요).");
                Console.WriteLine("설치 위치: " + Home);
                Directory.CreateDirectory(Home);
                if (!File.Exists(python)) InstallPython();
                if (!File.Exists(updater)) InstallApp();
            }
            // updater 가 라이브러리·모델 설치를 확인하고 프로그램을 띄움 (오류 시 직접 안내 후 대기)
            // -s: 이 PC 사용자 폴더(%APPDATA%\Python)의 다른 패키지가 섞이지 않게 함
            return Run(python, "-s \"" + updater + "\" --bootstrap", app);
        }
        catch (Exception e)
        {
            Console.WriteLine();
            Console.WriteLine("[오류] " + e.Message);
            Console.WriteLine("인터넷 연결을 확인한 뒤 다시 실행해 주세요.");
            Console.WriteLine("엔터를 누르면 종료합니다...");
            Console.ReadLine();
            return 1;
        }
    }

    static void InstallPython()
    {
        Console.WriteLine();
        Console.WriteLine("[1/2] Python 내려받는 중...");
        string zip = Download(PythonZipUrl);
        string tmpDir = Path.Combine(Home, "python_tmp");
        ExtractFresh(zip, tmpDir);
        File.Delete(zip);
        // embeddable Python 이 pip 로 설치한 라이브러리(Lib\site-packages)를 쓰도록 설정
        File.WriteAllText(Path.Combine(tmpDir, "python310._pth"),
                          "python310.zip\r\n.\r\nLib\\site-packages\r\nimport site\r\n");
        MoveFresh(tmpDir, Path.Combine(Home, "python"));
    }

    static void InstallApp()
    {
        Console.WriteLine();
        Console.WriteLine("[2/2] 프로그램 내려받는 중...");
        string zip = Download("https://codeload.github.com/" + Repo + "/zip/refs/heads/main");
        string tmpDir = Path.Combine(Home, "app_tmp");
        ExtractFresh(zip, tmpDir);
        File.Delete(zip);
        string[] roots = Directory.GetDirectories(tmpDir);  // cell-counter-main\
        if (roots.Length != 1) throw new Exception("내려받은 프로그램 압축 파일 구조가 예상과 다릅니다.");
        MoveFresh(roots[0], Path.Combine(Home, "app"));
        Directory.Delete(tmpDir, true);
    }

    static string Download(string url)
    {
        string dir = Path.Combine(Home, "_download");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, Guid.NewGuid().ToString("N") + ".zip");
        HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
        req.UserAgent = "CellCounter-Launcher";
        using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
        using (Stream src = resp.GetResponseStream())
        using (FileStream dst = File.Create(path))
        {
            long total = resp.ContentLength, done = 0;
            byte[] buf = new byte[1 << 16];
            int n, lastPct = -1;
            while ((n = src.Read(buf, 0, buf.Length)) > 0)
            {
                dst.Write(buf, 0, n);
                done += n;
                if (total > 0)
                {
                    int pct = (int)(done * 100 / total);
                    if (pct != lastPct) { Console.Write("\r  " + pct + "% (" + (done >> 20) + " / " + (total >> 20) + " MB)   "); lastPct = pct; }
                }
                else Console.Write("\r  " + (done >> 10) + " KB   ");
            }
        }
        Console.WriteLine();
        return path;
    }

    static void ExtractFresh(string zip, string dir)
    {
        if (Directory.Exists(dir)) Directory.Delete(dir, true);
        ZipFile.ExtractToDirectory(zip, dir);
    }

    static void MoveFresh(string src, string dst)
    {
        if (Directory.Exists(dst)) Directory.Delete(dst, true);
        Directory.Move(src, dst);
    }

    static int Run(string exe, string args, string cwd)
    {
        ProcessStartInfo psi = new ProcessStartInfo(exe, args);
        psi.UseShellExecute = false;
        psi.WorkingDirectory = cwd;
        psi.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
        using (Process p = Process.Start(psi))
        {
            p.WaitForExit();
            return p.ExitCode;
        }
    }
}
