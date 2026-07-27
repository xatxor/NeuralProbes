export const darkTheme = {
  kind: "dark" as const,
  text: {
    primary: "#e8e8e8",
    secondary: "#a8a8a8",
    tertiary: "#787878",
    quaternary: "#585858",
    onAccent: "#ffffff",
    link: "#4da3ff",
  },
  bg: {
    editor: "#1e1e1e",
    elevated: "#252526",
    chrome: "#181818",
  },
  fill: {
    primary: "#333333",
    secondary: "#2a2d2e",
    tertiary: "#232323",
    quaternary: "#1a1a1a",
  },
  stroke: {
    primary: "#3c3c3c",
    secondary: "#454545",
    tertiary: "#555555",
  },
  accent: {
    primary: "#4da3ff",
    control: "#0e639c",
  },
};

export type HostTheme = typeof darkTheme;

export function useHostTheme(): HostTheme {
  return darkTheme;
}
