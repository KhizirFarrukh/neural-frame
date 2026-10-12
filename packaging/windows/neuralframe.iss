; Inno Setup script for the NeuralFrame Windows installer.
; Compiled by packaging/windows/build_release.py, which defines AppVersion, SourceDir, OutputDir, IconFile
; and Spanning on the ISCC command line.
; Per-user install (no administrator rights): %LOCALAPPDATA%\Programs\NeuralFrame.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Spanning
  #define Spanning "no"
#endif

[Setup]
AppId={{6B1E3C1A-2F4D-4E8B-9C3A-7D5E1F2A9B40}
AppName=NeuralFrame
AppVersion={#AppVersion}
AppVerName=NeuralFrame {#AppVersion}
AppPublisher=Khizir Farrukh
DefaultDirName={autopf}\NeuralFrame
DefaultGroupName=NeuralFrame
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=NeuralFrame-{#AppVersion}-win64-Setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\NeuralFrame.exe
UninstallDisplayName=NeuralFrame {#AppVersion}
InfoBeforeFile={#SourceDir}\README.txt
Compression=lzma2/normal
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
DiskSpanning={#Spanning}
DiskSliceSize=max
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\NeuralFrame"; Filename: "{app}\NeuralFrame.exe"
Name: "{autoprograms}\NeuralFrame command line"; Filename: "{cmd}"; Parameters: "/k cd /d ""{app}"" && neuralframe-cli.exe --help"; WorkingDir: "{app}"
Name: "{autoprograms}\NeuralFrame read me"; Filename: "{app}\README.txt"
Name: "{autodesktop}\NeuralFrame"; Filename: "{app}\NeuralFrame.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\NeuralFrame.exe"; Description: "{cm:LaunchProgram,NeuralFrame}"; Flags: nowait postinstall skipifsilent
